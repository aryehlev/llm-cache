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
        }
    }
}

/// A cache-management action for the provider adapter to execute.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Create a provider-side cache for the prefix ending at `node`.
    Create {
        /// Trie node whose prefix should be cached.
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
}

/// The PCOE engine. One instance per (provider, model) — provider caches are
/// model-scoped, so traffic to different models must not share a trie.
pub struct Engine {
    trie: PrefixTrie,
    prices: PriceSheet,
    cfg: Config,
    cached: BTreeSet<NodeId>,
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
        let covered_tokens = deepest_cached.map_or(0, |id| self.trie.tokens(id));

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
        {
            let node = self.trie.node_mut(id);
            node.state = CacheState::Cached {
                expires_at_hours: now + tau_hold,
                ttl_hours: tau_hold,
            };
            node.last_flip = now;
        }
        self.cached.insert(id);
        // Cached ancestors this deeper cache dominates are retired by the
        // exclusive-traffic check in `tick()` once the new node's rate
        // estimator warms up enough to attribute the traffic correctly.
    }

    /// Periodic maintenance: ski-rental deletions, TTL extensions, and trie GC.
    ///
    /// Call at least every [`Config::extend_lead_hours`] for extensions to land
    /// before provider-side expiry.
    pub fn tick(&mut self, now: f64) -> Vec<Action> {
        let mut actions = Vec::new();
        if let Some(tau_hold) = self.prices.tau_hold_hours() {
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
                let hold_until = node.stats.t_last + tau_hold;
                // Rate-informed early exit: with a confident estimator, if the
                // majority of observed gaps exceed the break-even hold time,
                // storage between hits costs more than recreation on average.
                let early = node.stats.samples >= self.cfg.confidence_samples
                    && node.stats.gap_fraction_gt(tau_hold) > 0.5;
                if now >= hold_until || early {
                    // Emit Delete even if the TTL just lapsed on its own (the
                    // controller extends exactly to the hold boundary, so the
                    // two coincide); provider deletes are idempotent.
                    actions.push(Action::Delete { node: id });
                    let node = self.trie.node_mut(id);
                    node.state = CacheState::Uncached;
                    node.last_flip = now;
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
        self.trie
            .gc(now, self.cfg.gc_min_lambda, self.cfg.gc_min_idle_hours);
        actions
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
    /// reverts the node to uncached so the state model stays truthful.
    pub fn mark_failed(&mut self, node: NodeId) {
        self.trie.node_mut(node).state = CacheState::Uncached;
        self.cached.remove(&node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefix(n: u64) -> Vec<u64> {
        (0..n).map(|i| 0x1000 + i).collect()
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
                eng.observe(&blocks, now);
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
    fn extends_before_expiry_while_traffic_flows() {
        let mut eng = Engine::new(PriceSheet::gemini_pro_like(), Config::default());
        let blocks = prefix(20);
        let mut extends = 0;
        // 2 hours of steady 5-minute traffic with 1-minute ticks.
        for m in 0..120u32 {
            let now = m as f64 / 60.0;
            if m % 5 == 0 {
                eng.observe(&blocks, now);
            }
            for a in eng.tick(now) {
                match a {
                    Action::Extend { .. } => extends += 1,
                    Action::Delete { .. } => panic!("must not delete under steady traffic"),
                    Action::Create { .. } => unreachable!("tick never creates"),
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
