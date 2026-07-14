//! The PCOE engine: observation intake plus the ski-rental lifecycle controller
//! (DESIGN.md §3.3) and plan commitment (§3.2).
//!
//! The engine never talks to a provider. It emits [`Action`]s; an adapter layer
//! executes them (`cachedContents.create/patch/delete` on Gemini, marker
//! placement on Anthropic). All engine decisions are advisory and fail-open: a
//! dropped action costs money at worst, never correctness.

use std::collections::{BTreeSet, HashMap};

use crate::plan::{cost_rate, plan_breakpoints, savings_rate, Placement};
use crate::price::{PriceSheet, Regime};
use crate::trie::{CacheState, NodeId, PrefixTrie};

/// Tunable engine parameters (defaults follow DESIGN.md §3).
#[derive(Clone, Debug)]
pub struct Config {
    /// Tokens per block; callers must chunk with the same value (see [`crate::chunk`]).
    pub block_tokens: u64,
    /// EWMA decay constant for rate estimates, hours.
    pub tau_decay_hours: f64,
    /// Promote-to-cached threshold: require `S > theta_up * C`.
    pub theta_up: f64,
    /// Demote threshold: keep while `S > theta_down * C`.
    pub theta_down: f64,
    /// Minimum time between state flips of one node, hours.
    pub min_dwell_hours: f64,
    /// Creation gate: expect at least `1 + margin` hits within the hold window.
    pub create_margin: f64,
    /// Decayed sample count above which the rate estimate is trusted for
    /// early (rate-informed) deletion.
    pub confidence_samples: f64,
    /// Emit an Extend when a cache is within this lead of expiry, hours.
    /// Callers must tick at least this often for extensions to land in time.
    pub extend_lead_hours: f64,
    /// GC: remove uncached leaves whose decayed rate is below this...
    pub gc_min_lambda: f64,
    /// ...and which have been idle at least this long, hours.
    pub gc_min_idle_hours: f64,
    /// Hard cap on trie size; beyond it, unseen suffixes are not materialized.
    pub max_nodes: usize,
    /// Enable write-amortizing micro-batching (DESIGN.md §3.4b): followers of
    /// an in-flight cache write receive a bounded defer advice so one write
    /// premium is amortized across the batch. Off by default (adds latency).
    pub enable_micro_batch: bool,
    /// Maximum defer for a follower, hours (default 2 seconds).
    pub micro_batch_defer_hours: f64,
    /// A Create left unconfirmed this long is presumed failed and reverted.
    pub pending_timeout_hours: f64,
    /// Enable predictive pre-creation ahead of a confidently periodic return
    /// (DESIGN.md §3.3). On by default; the strict confidence gate below means
    /// it only ever fires on genuinely periodic prefixes (daily opens, per-
    /// tenant windows), so unpredictable traffic is untouched.
    pub enable_predictive: bool,
    /// Lead time to pre-create before the predicted return, hours.
    pub predictive_lead_hours: f64,
    /// Minimum consistent return gaps before a period is trusted.
    pub predictive_min_returns: u32,
    /// Maximum coefficient of variation of the return gap to still call it
    /// periodic (smaller = stricter; a metronome-regular daily open is ~0).
    /// Genuinely periodic prefixes (daily opens, per-tenant windows) measure
    /// CV ≲ 0.01; merely bursty traffic that briefly looks regular measures
    /// CV ≳ 0.08, so this sits an order of magnitude below the noise floor to
    /// keep pre-creation strictly off unpredictable traffic (zero-regression).
    pub predictive_max_cv: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            block_tokens: 256,
            tau_decay_hours: 1.0,
            theta_up: 1.2,
            theta_down: 0.8,
            min_dwell_hours: 5.0 / 60.0,
            create_margin: 0.05,
            confidence_samples: 5.0,
            extend_lead_hours: 5.0 / 60.0,
            gc_min_lambda: 0.01,
            gc_min_idle_hours: 24.0,
            max_nodes: 200_000,
            enable_micro_batch: false,
            micro_batch_defer_hours: 2.0 / 3600.0,
            pending_timeout_hours: 3.0 / 60.0,
            enable_predictive: true,
            predictive_lead_hours: 5.0 / 60.0,
            predictive_min_returns: 3,
            predictive_max_cv: 0.05,
        }
    }
}

/// A cache-management action for the provider adapter to execute.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Create a provider-side cache for the prefix ending at `node`, using the
    /// content of the request being observed (the common case).
    Create {
        /// Trie node whose prefix should be cached.
        node: NodeId,
        /// Prefix length in tokens.
        tokens: u64,
        /// Requested initial TTL, hours.
        ttl_hours: f64,
    },
    /// **Pre-create** a cache ahead of a *predicted* periodic return, before
    /// any request arrives (DESIGN.md §3.3, predictive hold). Because there is
    /// no in-flight request to take content from, the application must serve
    /// this from its retained stable-prefix content (a fixed system prompt /
    /// corpus) — hence a distinct action from [`Action::Create`]. Emitted only
    /// from [`Engine::tick`], only for prefixes with a confidently periodic
    /// return, so it never fires on unpredictable traffic.
    PreCreate {
        /// Trie node whose prefix should be pre-created.
        node: NodeId,
        /// Prefix length in tokens.
        tokens: u64,
        /// Requested initial TTL, hours.
        ttl_hours: f64,
    },
    /// Extend a live cache's expiry (Gemini `expire_time` update).
    Extend {
        /// The cached node.
        node: NodeId,
        /// New expiry time, hours on the caller's clock.
        expires_at_hours: f64,
    },
    /// Delete a live cache.
    Delete {
        /// The cached node.
        node: NodeId,
    },
}

/// Micro-batching advice: this request's prefix has a cache write in flight;
/// briefly deferring dispatch lets it read the cache the leader is writing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Defer {
    /// The pending node the request would read from once confirmed.
    pub node: NodeId,
    /// Latest time to wait until, hours; dispatch uncached at this deadline if
    /// the create is still unconfirmed. Bounds added latency.
    pub until_hours: f64,
}

/// Result of observing one outgoing request.
#[derive(Clone, Debug)]
pub struct Observation {
    /// Trie nodes along the request's block path (may be shorter than the
    /// request if the node budget truncated it).
    pub path: Vec<NodeId>,
    /// Deepest node on the path with a live provider-side cache — the cache the
    /// adapter should reference for this request.
    pub deepest_cached: Option<NodeId>,
    /// Prefix tokens covered by `deepest_cached` (0 when none).
    pub cached_tokens: u64,
    /// Lifecycle actions triggered by this observation (creations, and
    /// deletions of newly-dominated ancestors).
    pub actions: Vec<Action>,
    /// Micro-batching advice ([`Config::enable_micro_batch`]); `None` unless a
    /// cache write for this prefix is currently in flight.
    pub defer: Option<Defer>,
}

/// The PCOE engine. One instance per (provider, model) — provider caches are
/// model-scoped, so traffic to different models must not share a trie.
pub struct Engine {
    trie: PrefixTrie,
    prices: PriceSheet,
    cfg: Config,
    cached: BTreeSet<NodeId>,
    pending: BTreeSet<NodeId>,
}

impl Engine {
    /// Create an engine for one provider price sheet.
    pub fn new(prices: PriceSheet, cfg: Config) -> Self {
        let trie = PrefixTrie::new(cfg.block_tokens, cfg.tau_decay_hours, cfg.max_nodes);
        Engine {
            trie,
            prices,
            cfg,
            cached: BTreeSet::new(),
            pending: BTreeSet::new(),
        }
    }

    /// The price sheet this engine optimizes against.
    pub fn prices(&self) -> &PriceSheet {
        &self.prices
    }

    /// Read access to the traffic model.
    pub fn trie(&self) -> &PrefixTrie {
        &self.trie
    }

    /// Nodes the engine currently believes have live provider-side caches.
    pub fn cached_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.cached.iter().copied()
    }

    /// Record one outgoing request (as block hashes; see [`crate::chunk`]) at
    /// time `now`, returning the cache to reference and any lifecycle actions.
    ///
    /// Never blocks the request path: creations are fire-and-forget and the
    /// observed request itself proceeds however the current cache state allows.
    pub fn observe(&mut self, blocks: &[u64], now: f64) -> Observation {
        let path = self.trie.observe_path(blocks, now);

        // Return feedback (storage regime): the first request after a delete
        // tells us how long the prefix was actually idle. That drives two
        // learners, both keyed off `awaiting_return` so each delete is counted
        // once (DESIGN.md §3.3):
        //   1. Learned hold — if the return came inside tau_effective the
        //      delete was premature; grow the hold to ride through such lulls.
        //   2. Periodicity — record the return gap; a prefix that reliably
        //      returns after ~the same idle can be pre-created just before it.
        if let (Some(tau_hold), Some(tau_eff)) = (
            self.prices.tau_hold_hours(),
            self.prices.tau_effective_hours(),
        ) {
            for &id in &path {
                let node = self.trie.node(id);
                if !node.awaiting_return || !node.last_delete.is_finite() {
                    continue;
                }
                let since_delete = (now - node.last_delete).max(0.0);
                let prior_hold = node.hold_hint.max(tau_hold);
                let n = self.trie.node_mut(id);
                if since_delete < tau_eff {
                    // Premature: the gap that beat us = held idle + since delete.
                    n.hold_hint = (prior_hold + since_delete).min(tau_eff);
                }
                // Periodicity: only learn from *genuine* idle-and-return cycles
                // — gaps past `tau_eff`, where the prefix definitely went cold
                // and came back. Brief intraday lulls (below `tau_eff`) are the
                // learned-hold's job, not pre-creation's; feeding them here would
                // mix minute-scale and hour-scale gaps into one estimator and
                // inflate its variance past the periodicity gate. The gap is
                // measured from the *delete* (when the node started awaiting), so
                // pre-creation lands just before the actual return.
                if since_delete >= tau_eff {
                    n.record_return_gap(since_delete);
                }
                n.awaiting_return = false;
            }
        }

        // Deepest live cached node; lazily clear entries the provider has
        // already expired.
        let mut deepest_cached = None;
        for &id in &path {
            if let CacheState::Cached {
                expires_at_hours, ..
            } = self.trie.node(id).state
            {
                if expires_at_hours > now {
                    deepest_cached = Some(id);
                } else {
                    self.trie.node_mut(id).state = CacheState::Uncached;
                    self.cached.remove(&id);
                }
            }
        }

        // Micro-batching: computed before any creation this observation might
        // trigger, so a leader never defers on its own write.
        let mut defer = None;
        if self.cfg.enable_micro_batch && self.prices.regime() == Regime::StorageMetered {
            for &id in path.iter().rev() {
                if let CacheState::Pending { since_hours } = self.trie.node(id).state {
                    if now - since_hours < self.cfg.micro_batch_defer_hours {
                        defer = Some(Defer {
                            node: id,
                            until_hours: since_hours + self.cfg.micro_batch_defer_hours,
                        });
                    }
                    break;
                }
            }
        }

        let mut actions = Vec::new();
        if self.prices.regime() == Regime::StorageMetered {
            self.consider_create(&path, deepest_cached, now, &mut actions);
            // A creation may have changed the deepest cached node for *future*
            // requests, but this request still references the pre-existing one.
        }

        let cached_tokens = deepest_cached.map_or(0, |id| self.trie.tokens(id));
        Observation {
            path,
            deepest_cached,
            cached_tokens,
            actions,
            defer,
        }
    }

    /// Storage-regime creation rule (DESIGN.md §3.3): cache the deepest
    /// eligible node on the path expected to earn back its creation cost
    /// within the hold window.
    fn consider_create(
        &mut self,
        path: &[NodeId],
        deepest_cached: Option<NodeId>,
        now: f64,
        actions: &mut Vec<Action>,
    ) {
        let Some(tau_hold) = self.prices.tau_hold_hours() else {
            return;
        };
        // In-flight (Pending) creates count as coverage too: while one is in
        // flight, a shallower create for the same traffic is pure waste.
        let pending_tokens = path
            .iter()
            .rev()
            .find(|&&id| matches!(self.trie.node(id).state, CacheState::Pending { .. }))
            .map_or(0, |&id| self.trie.tokens(id));
        let covered_tokens = deepest_cached
            .map_or(0, |id| self.trie.tokens(id))
            .max(pending_tokens);

        let mut chosen = None;
        for &id in path.iter().rev() {
            let tokens = self.trie.tokens(id);
            if tokens < self.prices.min_cacheable_tokens {
                // Tokens shrink toward the root; nothing shallower qualifies.
                break;
            }
            let node = self.trie.node(id);
            if node.state != CacheState::Uncached {
                continue;
            }
            if now - node.last_flip < self.cfg.min_dwell_hours {
                continue;
            }
            if tokens <= covered_tokens {
                // A live cache already covers at least this much prefix.
                break;
            }
            let lambda = node.stats.decayed_lambda(now, self.cfg.tau_decay_hours);
            if lambda * tau_hold < 1.0 + self.cfg.create_margin {
                continue; // won't earn a hit back inside the hold window
            }
            let (c, _) = cost_rate(
                &self.prices,
                tokens,
                &node.stats,
                now,
                self.cfg.tau_decay_hours,
            );
            let s = savings_rate(&self.prices, tokens, lambda);
            if s > self.cfg.theta_up * c {
                chosen = Some(id);
                break; // deepest eligible wins
            }
        }

        let Some(id) = chosen else { return };
        let tokens = self.trie.tokens(id);
        actions.push(Action::Create {
            node: id,
            tokens,
            ttl_hours: tau_hold,
        });
        // The node stays Pending until the adapter confirms the provider-side
        // create ([`Engine::confirm_create`]) or reports failure
        // ([`Engine::mark_failed`]); a timeout in `tick()` is the backstop.
        {
            let node = self.trie.node_mut(id);
            node.state = CacheState::Pending { since_hours: now };
            node.last_flip = now;
        }
        self.pending.insert(id);
        // Cached ancestors this deeper cache dominates are retired by the
        // exclusive-traffic check in `tick()` once the new node's rate
        // estimator warms up enough to attribute the traffic correctly.
    }

    /// Confirm a provider-side create succeeded: the node's cache is live from
    /// `now` for the TTL requested in the Create action.
    pub fn confirm_create(&mut self, node: NodeId, now: f64) {
        let Some(tau_hold) = self.prices.tau_hold_hours() else {
            return;
        };
        if matches!(self.trie.node(node).state, CacheState::Pending { .. }) {
            let n = self.trie.node_mut(node);
            n.state = CacheState::Cached {
                expires_at_hours: now + tau_hold,
                ttl_hours: tau_hold,
            };
            self.pending.remove(&node);
            self.cached.insert(node);
        }
    }

    /// Periodic maintenance: ski-rental deletions, TTL extensions, and trie GC.
    ///
    /// Call at least every [`Config::extend_lead_hours`] for extensions to land
    /// before provider-side expiry.
    pub fn tick(&mut self, now: f64) -> Vec<Action> {
        let mut actions = Vec::new();
        // Presume unconfirmed creates failed after the timeout (fail-open: the
        // node just becomes creatable again after the dwell).
        for id in self.pending.clone() {
            match self.trie.node(id).state {
                CacheState::Pending { since_hours }
                    if now - since_hours > self.cfg.pending_timeout_hours =>
                {
                    let n = self.trie.node_mut(id);
                    n.state = CacheState::Uncached;
                    n.last_flip = now;
                    self.pending.remove(&id);
                }
                CacheState::Pending { .. } => {}
                _ => {
                    self.pending.remove(&id);
                }
            }
        }
        if let Some(tau_hold) = self.prices.tau_hold_hours() {
            let tau_eff = self.prices.tau_effective_hours().unwrap_or(tau_hold);
            self.retire_dominated(now, tau_hold, &mut actions);
            for id in self.cached.clone() {
                let node = self.trie.node(id);
                let CacheState::Cached {
                    expires_at_hours,
                    ttl_hours,
                } = node.state
                else {
                    self.cached.remove(&id);
                    continue;
                };
                // Per-node hold: the learned hint when this prefix has shown it
                // recurs after a lull, else the conservative floor; never past
                // tau_effective.
                let node_hold = if node.hold_hint > tau_hold {
                    node.hold_hint.min(tau_eff)
                } else {
                    tau_hold
                };
                let hold_until = node.stats.t_last + node_hold;
                // Rate-informed early exit: with a confident estimator, if the
                // majority of observed gaps exceed the break-even hold time,
                // storage between hits costs more than recreation on average.
                let early = node.stats.samples >= self.cfg.confidence_samples
                    && node.stats.gap_fraction_gt(node_hold) > 0.5;
                if now >= hold_until || early {
                    // Emit Delete even if the TTL just lapsed on its own (the
                    // controller extends exactly to the hold boundary, so the
                    // two coincide); provider deletes are idempotent. Record the
                    // delete time so a quick return can be recognized as
                    // premature (learned-hold feedback in `observe`).
                    actions.push(Action::Delete { node: id });
                    let node = self.trie.node_mut(id);
                    node.state = CacheState::Uncached;
                    node.last_flip = now;
                    node.last_delete = now;
                    node.awaiting_return = true;
                    self.cached.remove(&id);
                } else if expires_at_hours <= now {
                    // Extension cadence was missed and the provider expired the
                    // entry mid-hold; nothing to delete remotely.
                    let node = self.trie.node_mut(id);
                    node.state = CacheState::Uncached;
                    node.last_flip = now;
                    self.cached.remove(&id);
                } else if expires_at_hours + 1e-9 < hold_until
                    && expires_at_hours - now <= self.cfg.extend_lead_hours
                {
                    actions.push(Action::Extend {
                        node: id,
                        expires_at_hours: hold_until,
                    });
                    self.trie.node_mut(id).state = CacheState::Cached {
                        expires_at_hours: hold_until,
                        ttl_hours,
                    };
                }
            }
        }
        self.emit_predictive_precreates(now, &mut actions);
        self.trie
            .gc(now, self.cfg.gc_min_lambda, self.cfg.gc_min_idle_hours);
        actions
    }

    /// Predictive pre-creation (DESIGN.md §3.3): for a prefix whose post-delete
    /// returns are confidently periodic, pre-create the cache a short lead
    /// before the predicted next return, so the first request of the next
    /// active window reads cached instead of paying the cold-start miss (the
    /// dominant PCOE-vs-oracle gap on per-tenant fleets). The strict
    /// confidence gate (min returns + low variation) means only genuinely
    /// periodic prefixes ever pre-create; bursty and one-off traffic never do.
    fn emit_predictive_precreates(&mut self, now: f64, actions: &mut Vec<Action>) {
        if !self.cfg.enable_predictive {
            return;
        }
        let Some(tau_hold) = self.prices.tau_hold_hours() else {
            return;
        };
        // Candidates: nodes currently idle after a delete, awaiting their
        // predicted return.
        let candidates: Vec<NodeId> = self
            .trie
            .awaiting_return_nodes()
            .into_iter()
            .filter(|&id| matches!(self.trie.node(id).state, CacheState::Uncached))
            .collect();
        for id in candidates {
            let node = self.trie.node(id);
            let Some((period, cv)) = node.return_prediction(self.cfg.predictive_min_returns) else {
                continue;
            };
            if cv > self.cfg.predictive_max_cv {
                continue; // not periodic enough to bet a write on
            }
            let predicted = node.last_delete + period;
            // Fire once we enter the lead window before the prediction, but not
            // absurdly early (a late request is fine; the cache just waits).
            if now >= predicted - self.cfg.predictive_lead_hours && now < predicted {
                let tokens = self.trie.tokens(id);
                if tokens < self.prices.min_cacheable_tokens {
                    continue;
                }
                actions.push(Action::PreCreate {
                    node: id,
                    tokens,
                    ttl_hours: tau_hold,
                });
                let n = self.trie.node_mut(id);
                n.state = CacheState::Pending { since_hours: now };
                n.last_flip = now;
                n.awaiting_return = false; // consumed; don't re-fire this cycle
                                           // Reset the hold clock so the freshly pre-created cache isn't
                                           // instantly deleted against its stale last-hit time.
                n.stats.t_last = now;
                self.pending.insert(id);
            }
        }
    }

    /// Exclusive-coverage retirement (DESIGN.md §3.2 semantics applied to the
    /// storage regime): a cached node whose traffic is almost entirely served
    /// by deeper cached descendants earns nothing on its own — its exclusive
    /// arrival rate no longer clears break-even — so its storage meter is pure
    /// waste. Attribute each cached node's rate to its *nearest* cached
    /// ancestor and retire ancestors whose exclusive remainder is below the
    /// creation gate.
    fn retire_dominated(&mut self, now: f64, tau_hold: f64, actions: &mut Vec<Action>) {
        if self.cached.len() < 2 {
            return;
        }
        let tau = self.cfg.tau_decay_hours;
        let snapshot: Vec<NodeId> = self.cached.iter().copied().collect();
        let mut covered: HashMap<NodeId, f64> = HashMap::new();
        for &b in &snapshot {
            let lambda_b = self.trie.node(b).stats.decayed_lambda(now, tau);
            let mut cur = self.trie.node(b).parent;
            while let Some(p) = cur {
                if self.cached.contains(&p) {
                    *covered.entry(p).or_insert(0.0) += lambda_b;
                    break;
                }
                cur = self.trie.node(p).parent;
            }
        }
        for (a, sub) in covered {
            let lam_excl = (self.trie.node(a).stats.decayed_lambda(now, tau) - sub).max(0.0);
            if lam_excl * tau_hold < 1.0 + self.cfg.create_margin {
                actions.push(Action::Delete { node: a });
                let node = self.trie.node_mut(a);
                node.state = CacheState::Uncached;
                node.last_flip = now;
                self.cached.remove(&a);
            }
        }
    }

    /// Compute the current optimal breakpoint placement (DESIGN.md §3.2). On
    /// the write-premium regime the plan is committed to the trie state (with
    /// hysteresis and dwell) so subsequent [`Engine::observe`] calls report the
    /// placed nodes via `deepest_cached`; on other regimes it is advisory.
    pub fn plan(&mut self, now: f64) -> Vec<Placement> {
        let placements = plan_breakpoints(
            &self.trie,
            &self.prices,
            now,
            self.cfg.theta_up,
            self.cfg.theta_down,
        );
        if self.prices.regime() == Regime::WritePremium {
            let chosen: BTreeSet<NodeId> = placements.iter().map(|p| p.node).collect();
            for id in self.cached.clone() {
                if !chosen.contains(&id)
                    && now - self.trie.node(id).last_flip >= self.cfg.min_dwell_hours
                {
                    let node = self.trie.node_mut(id);
                    node.state = CacheState::Uncached;
                    node.last_flip = now;
                    self.cached.remove(&id);
                }
            }
            for p in &placements {
                if !self.cached.contains(&p.node) {
                    let node = self.trie.node_mut(p.node);
                    node.state = CacheState::Cached {
                        // Write-premium entries are refreshed free on read;
                        // lifetime is not managed by the controller.
                        expires_at_hours: f64::INFINITY,
                        ttl_hours: p.ttl_hours.unwrap_or(0.0),
                    };
                    node.last_flip = now;
                    self.cached.insert(p.node);
                }
            }
        }
        placements
    }

    /// Tell the engine a provider-side action failed (e.g. create rejected):
    /// reverts the node to uncached so the state model stays truthful. The
    /// dwell timer restarts, giving the provider a breather before a retry.
    pub fn mark_failed(&mut self, node: NodeId, now: f64) {
        let n = self.trie.node_mut(node);
        n.state = CacheState::Uncached;
        n.last_flip = now;
        self.cached.remove(&node);
        self.pending.remove(&node);
    }

    /// Estimate the input-token cost of sending this request through this
    /// engine's provider right now, given its live cache state — the quantity
    /// cross-provider routing compares (DESIGN.md §3.4c). Read-only: does not
    /// record a traversal.
    pub fn quote_input_cost(&self, blocks: &[u64], total_tokens: u64, now: f64) -> f64 {
        let path = self.trie.peek_path(blocks);
        let mut cached = 0u64;
        for &id in &path {
            if let CacheState::Cached {
                expires_at_hours, ..
            } = self.trie.node(id).state
            {
                if expires_at_hours > now {
                    cached = self.trie.tokens(id);
                }
            }
        }
        let cached = cached.min(total_tokens);
        cached as f64 / 1e6 * self.prices.cached_read_per_mtok
            + (total_tokens - cached) as f64 / 1e6 * self.prices.input_per_mtok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefix(n: u64) -> Vec<u64> {
        (0..n).map(|i| 0x1000 + i).collect()
    }

    /// Simulate an instantly-successful adapter: confirm every Create.
    fn confirm_creates(eng: &mut Engine, actions: &[Action], now: f64) {
        for a in actions {
            if let Action::Create { node, .. } = a {
                eng.confirm_create(*node, now);
            }
        }
    }

    #[test]
    fn creates_once_rate_clears_breakeven_and_reads_after() {
        let mut eng = Engine::new(PriceSheet::gemini_pro_like(), Config::default());
        let blocks = prefix(20); // 5120 tokens >= 4096 minimum
        let mut creates = 0;
        let mut cached_reads = 0;
        // 45 minutes of 5-minute traffic, ticking every minute so extensions
        // keep the entry alive.
        for m in 0..=45u32 {
            let now = m as f64 / 60.0;
            if m % 5 == 0 {
                let obs = eng.observe(&blocks, now);
                creates += obs
                    .actions
                    .iter()
                    .filter(|a| matches!(a, Action::Create { .. }))
                    .count();
                if obs.deepest_cached.is_some() {
                    cached_reads += 1;
                    assert_eq!(obs.cached_tokens, 5120);
                }
                confirm_creates(&mut eng, &obs.actions, now);
            }
            eng.tick(now);
        }
        assert_eq!(creates, 1, "hysteresis must prevent repeated creation");
        assert!(cached_reads >= 5, "cached reads: {cached_reads}");
    }

    #[test]
    fn ski_rental_deletes_tau_hold_after_last_hit() {
        let prices = PriceSheet::gemini_pro_like();
        let tau_hold = prices.tau_hold_hours().unwrap();
        let mut eng = Engine::new(prices, Config::default());
        let blocks = prefix(20);
        let mut last_hit = 0.0;
        let mut deleted_at = None;
        // Traffic every 5 minutes until minute 45, then silence; 1-minute ticks.
        for m in 0..=120u32 {
            let now = m as f64 / 60.0;
            if m % 5 == 0 && m <= 45 {
                let obs = eng.observe(&blocks, now);
                confirm_creates(&mut eng, &obs.actions, now);
                last_hit = now;
            }
            for a in eng.tick(now) {
                if matches!(a, Action::Delete { .. }) {
                    assert!(deleted_at.is_none(), "must delete exactly once");
                    deleted_at = Some(now);
                }
            }
        }
        let deleted_at = deleted_at.expect("idle cache must be deleted");
        assert!(deleted_at >= last_hit + tau_hold - 1e-9);
        assert!(deleted_at <= last_hit + tau_hold + 2.0 / 60.0);
        assert_eq!(eng.cached_nodes().count(), 0);
    }

    #[test]
    fn predictive_precreate_fires_and_warms_a_periodic_return() {
        // A prefix that is active, goes idle for a fixed period well past
        // tau_effective, and returns — cycle after cycle — is exactly what
        // predictive pre-creation targets. After enough clean cycles to learn
        // the period (>= predictive_min_returns), a tick in the lead window
        // before the next return must emit a PreCreate, and the first request of
        // that next window must then read warm (cached) instead of cold.
        let prices = PriceSheet::gemini_pro_like();
        let tau_eff = prices.tau_effective_hours().unwrap();
        let mut eng = Engine::new(prices, Config::default());
        let blocks = prefix(20);
        let period = 2.0_f64; // > tau_effective, so each idle is a real cycle

        let mut precreated_cycle: Option<u32> = None;
        let mut warm_after_precreate = false;
        for cycle in 0..7u32 {
            let base = cycle as f64 * period;
            // Active burst: a dozen minutes of dense traffic creates & confirms.
            for m in 0..12u32 {
                let now = base + m as f64 / 60.0;
                let obs = eng.observe(&blocks, now);
                // The first request of a window that a pre-create warmed reads
                // cached — that is the payoff we are checking.
                if m == 0
                    && cycle > 0
                    && precreated_cycle == Some(cycle - 1)
                    && obs.cached_tokens > 0
                {
                    warm_after_precreate = true;
                }
                confirm_creates(&mut eng, &obs.actions, now);
                eng.tick(now);
            }
            // Idle until the next window, ticking each minute; watch for a
            // pre-create and confirm it like a real adapter would.
            let mut t = base + 12.0 / 60.0;
            let next_base = (cycle as f64 + 1.0) * period;
            while t < next_base {
                for a in eng.tick(t) {
                    if let Action::PreCreate { node, .. } = a {
                        precreated_cycle.get_or_insert(cycle);
                        eng.confirm_create(node, t);
                    }
                }
                t += 1.0 / 60.0;
            }
        }

        let c = precreated_cycle.expect("a periodic return must eventually pre-create");
        // Never before the periodicity is trusted (>= min_returns clean cycles).
        assert!(
            c >= Config::default().predictive_min_returns,
            "pre-create fired too early, at cycle {c}"
        );
        assert!(
            warm_after_precreate,
            "the pre-created cache must warm the next window's first request"
        );
        // Sanity: the learned period is the idle gap (~period - tau_hold), well
        // past tau_effective, so this was a genuine idle-and-return cycle.
        assert!(tau_eff < period);
    }

    #[test]
    fn learned_hold_grows_after_a_premature_delete() {
        // Warm a prefix, let it delete after tau_hold, then send a request
        // just after the delete — a premature delete. The node's hold_hint
        // must grow so it will next hold longer, and a genuinely-idle prefix
        // (no quick return) must keep hold_hint at zero.
        let prices = PriceSheet::gemini_pro_like();
        let tau_hold = prices.tau_hold_hours().unwrap();
        let tau_eff = prices.tau_effective_hours().unwrap();
        let mut eng = Engine::new(prices, Config::default());
        let blocks = prefix(20);

        // Active burst so the cache is created and confirmed.
        let mut node = None;
        for m in 0..30u32 {
            let now = m as f64 / 60.0;
            let obs = eng.observe(&blocks, now);
            for a in &obs.actions {
                if let Action::Create { node: n, .. } = a {
                    eng.confirm_create(*n, now);
                    node = Some(*n);
                }
            }
            eng.tick(now);
        }
        let node = node.expect("cache created");
        assert_eq!(eng.trie().node(node).hold_hint, 0.0, "no premature yet");

        // Go idle just past tau_hold so it deletes, then return right after.
        let last = 29.0 / 60.0;
        let mut deleted_at = None;
        for k in 1..=40u32 {
            let now = last + k as f64 / 60.0;
            for a in eng.tick(now) {
                if matches!(a, Action::Delete { .. }) {
                    deleted_at = Some(now);
                }
            }
            if deleted_at.is_some() {
                break;
            }
        }
        let deleted_at = deleted_at.expect("must delete after tau_hold");
        // A request one minute after the delete = premature return.
        eng.observe(&blocks, deleted_at + 1.0 / 60.0);
        let hint = eng.trie().node(node).hold_hint;
        assert!(
            hint > tau_hold,
            "hold_hint {hint} must grow past tau_hold {tau_hold} after a premature delete"
        );
        assert!(
            hint <= tau_eff + 1e-9,
            "hold_hint must stay bounded by tau_effective"
        );
    }

    #[test]
    fn extends_before_expiry_while_traffic_flows() {
        let mut eng = Engine::new(PriceSheet::gemini_pro_like(), Config::default());
        let blocks = prefix(20);
        let mut extends = 0;
        // 2 hours of steady 5-minute traffic with 1-minute ticks.
        for m in 0..120u32 {
            let now = m as f64 / 60.0;
            if m % 5 == 0 {
                let obs = eng.observe(&blocks, now);
                confirm_creates(&mut eng, &obs.actions, now);
            }
            for a in eng.tick(now) {
                match a {
                    Action::Extend { .. } => extends += 1,
                    Action::Delete { .. } => panic!("must not delete under steady traffic"),
                    Action::Create { .. } => unreachable!("tick never creates"),
                    Action::PreCreate { .. } => {
                        panic!("steady traffic never deletes, so it never pre-creates")
                    }
                }
            }
        }
        assert!(extends > 0, "expiry must be pushed out while traffic flows");
    }

    #[test]
    fn deeper_create_retires_dominated_ancestor() {
        let prices = PriceSheet::gemini_pro_like();
        let mut eng = Engine::new(prices, Config::default());
        let short = prefix(20); // 5120 tokens
        let long: Vec<u64> = prefix(20).into_iter().chain(200..216).collect(); // 9216 tokens

        let mut shallow = None;
        let mut deep_created = false;
        let mut shallow_deleted = false;
        for m in 0..=300u32 {
            let now = m as f64 / 60.0;
            if m % 5 == 0 {
                // First half hour: short-prefix traffic; then the workload
                // switches to a longer shared prefix.
                let obs = if m < 30 {
                    eng.observe(&short, now)
                } else {
                    eng.observe(&long, now)
                };
                for a in &obs.actions {
                    match a {
                        Action::Create {
                            node, tokens: 5120, ..
                        } => shallow = Some(*node),
                        Action::Create { tokens: 9216, .. } => deep_created = true,
                        _ => {}
                    }
                }
                confirm_creates(&mut eng, &obs.actions, now);
            }
            for a in eng.tick(now) {
                if let Action::Delete { node } = a {
                    if Some(node) == shallow {
                        shallow_deleted = true;
                    }
                }
            }
        }
        assert!(shallow.is_some() && deep_created);
        assert!(
            shallow_deleted,
            "shallow cache with no exclusive traffic must be retired"
        );
        // Exactly the deep node remains cached (the covered-prefix guard must
        // prevent re-creating the dominated shallow node).
        let cached: Vec<_> = eng.cached_nodes().collect();
        assert_eq!(cached.len(), 1);
        assert_eq!(eng.trie().tokens(cached[0]), 9216);
    }

    #[test]
    fn micro_batch_defers_followers_until_create_confirms() {
        let cfg = Config {
            enable_micro_batch: true,
            ..Config::default()
        };
        let mut eng = Engine::new(PriceSheet::gemini_pro_like(), cfg);
        let blocks = prefix(20);

        // Warm until a create fires; the leader itself gets no defer.
        let mut created = None;
        let mut t = 0.0;
        for i in 0..8 {
            t = i as f64 * (5.0 / 60.0);
            let obs = eng.observe(&blocks, t);
            assert!(
                obs.defer.is_none(),
                "leader must not defer on its own write"
            );
            for a in &obs.actions {
                if let Action::Create { node, .. } = a {
                    created = Some(*node);
                }
            }
            if created.is_some() {
                break;
            }
        }
        let node = created.expect("creation must fire");

        // A follower 0.5s later sees the write in flight and is told to wait.
        let obs = eng.observe(&blocks, t + 0.5 / 3600.0);
        let defer = obs.defer.expect("follower must be deferred");
        assert_eq!(defer.node, node);
        assert!(obs.deepest_cached.is_none());
        assert!(defer.until_hours > t && defer.until_hours <= t + 2.1 / 3600.0);

        // The adapter confirms 1s after the create; the follower retries at
        // the advised deadline and reads the cache.
        eng.confirm_create(node, t + 1.0 / 3600.0);
        let obs = eng.observe(&blocks, defer.until_hours);
        assert!(obs.defer.is_none());
        assert_eq!(obs.deepest_cached, Some(node));
    }

    #[test]
    fn unconfirmed_create_times_out_and_retries_later() {
        let mut eng = Engine::new(PriceSheet::gemini_pro_like(), Config::default());
        let blocks = prefix(20);
        let mut creates = 0;
        // Never confirm: the adapter is "down". The engine must revert the
        // pending node after the timeout and try again after the dwell.
        for m in 0..=60u32 {
            let now = m as f64 / 60.0;
            if m % 5 == 0 {
                let obs = eng.observe(&blocks, now);
                creates += obs
                    .actions
                    .iter()
                    .filter(|a| matches!(a, Action::Create { .. }))
                    .count();
            }
            eng.tick(now);
        }
        assert!(creates >= 2, "creation must be retried, got {creates}");
        assert_eq!(
            eng.cached_nodes().count(),
            0,
            "nothing confirmed, nothing cached"
        );
    }

    #[test]
    fn write_premium_plan_commits_and_observe_reports_marker() {
        let mut eng = Engine::new(PriceSheet::anthropic_sonnet_like(), Config::default());
        let blocks = prefix(10); // 2560 tokens >= 2048 minimum
        for i in 0..30 {
            eng.observe(&blocks, i as f64 * (2.0 / 60.0));
        }
        let placements = eng.plan(1.0);
        assert!(!placements.is_empty());
        let obs = eng.observe(&blocks, 1.01);
        assert_eq!(obs.deepest_cached, Some(placements[0].node));
    }
}
