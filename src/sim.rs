//! ROI simulator: replay a traffic trace through PCOE and the alternatives,
//! and report the actual dollars each would have spent (DESIGN.md §8).
//!
//! This is the product's proof engine. Point it at a captured production trace
//! (privacy-safe — block hashes only) or a synthetic workload from
//! [`crate::trace`], and it answers the only question that matters: *how much
//! would PCOE have saved on this traffic, and how close to optimal is it?*
//!
//! Policies compared (storage-metered / Gemini-style regime):
//!
//! - **NoCache** — pay full input price on every request (the floor).
//! - **CacheEverything** — cache each shared prefix on first sight, hold
//!   forever (the naive strawman most teams reach for).
//! - **StaticTtl(h)** — cache-everything but expire an idle prefix after a
//!   fixed hand-set TTL (the best a human typically does by hand).
//! - **Pcoe** — the real [`crate::Engine`], with an instantly-successful
//!   adapter (creates confirmed immediately).
//! - **ReferenceOracle** — per-family offline-optimal ski-rental, computed by
//!   DP with perfect hindsight. The benchmark PCOE's competitive ratio is
//!   measured against.
//!
//! All policies pay the same per-request suffix cost (the non-cacheable tail),
//! so reported totals are real spend and differences are attributable purely
//! to cache management.

use std::collections::HashMap;

use crate::engine::{Action, Config, Engine};
use crate::price::PriceSheet;
use crate::trace::Event;
use crate::trie::{NodeId, PrefixTrie};

/// A policy to simulate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Policy {
    /// Never cache.
    NoCache,
    /// Cache each shared prefix on first sight; never delete.
    CacheEverything,
    /// Cache-everything with a fixed idle-TTL (hours) before deletion.
    StaticTtl(f64),
    /// The real PCOE engine.
    Pcoe,
    /// Per-family offline-optimal ski-rental (perfect hindsight benchmark).
    ReferenceOracle,
}

/// Cost breakdown and operational stats from one policy run.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// Total spend, dollars.
    pub total_cost: f64,
    /// Spend on cache creations (writes), dollars.
    pub create_cost: f64,
    /// Spend on storage (token-hours), dollars.
    pub storage_cost: f64,
    /// Spend on request tokens (cached reads + uncached input), dollars.
    pub request_cost: f64,
    /// Requests that read at least part of their prefix from cache.
    pub cached_requests: u64,
    /// Total requests.
    pub requests: u64,
    /// Cache create / extend / delete operation counts.
    pub creates: u64,
    /// Extend operations.
    pub extends: u64,
    /// Delete operations.
    pub deletes: u64,
    /// Cumulative token-hours held in cache.
    pub token_hours: f64,
}

impl Report {
    /// Fraction of requests served (partly) from cache.
    pub fn hit_rate(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.cached_requests as f64 / self.requests as f64
        }
    }

    /// Percent saved versus a baseline report.
    pub fn savings_vs(&self, baseline: &Report) -> f64 {
        if baseline.total_cost <= 0.0 {
            0.0
        } else {
            100.0 * (1.0 - self.total_cost / baseline.total_cost)
        }
    }
}

/// All policies run on one trace, plus derived ratios.
#[derive(Clone, Debug)]
pub struct Comparison {
    /// One report per policy, in a stable order.
    pub reports: Vec<(Policy, Report)>,
}

impl Comparison {
    /// The report for a policy, if present.
    pub fn get(&self, p: Policy) -> Option<&Report> {
        self.reports.iter().find(|(q, _)| *q == p).map(|(_, r)| r)
    }

    /// PCOE's competitive ratio: `pcoe_cost / oracle_cost` (1.0 = optimal).
    /// Note the oracle is a per-family reference, not a global optimum, so a
    /// ratio slightly below 1.0 is possible when PCOE exploits shared
    /// ancestors the per-family oracle does not.
    pub fn competitive_ratio(&self) -> Option<f64> {
        let pcoe = self.get(Policy::Pcoe)?.total_cost;
        let oracle = self.get(Policy::ReferenceOracle)?.total_cost;
        if oracle > 0.0 {
            Some(pcoe / oracle)
        } else {
            None
        }
    }
}

/// Run every policy on `trace` and return the comparison.
pub fn compare(trace: &[Event], prices: &PriceSheet, cfg: &Config) -> Comparison {
    let policies = [
        Policy::NoCache,
        Policy::CacheEverything,
        Policy::StaticTtl(1.0),
        Policy::Pcoe,
        Policy::ReferenceOracle,
    ];
    Comparison {
        reports: policies
            .iter()
            .map(|&p| (p, replay(trace, prices, cfg, p)))
            .collect(),
    }
}

/// Run a single policy on `trace`.
pub fn replay(trace: &[Event], prices: &PriceSheet, cfg: &Config, policy: Policy) -> Report {
    match policy {
        Policy::NoCache => replay_no_cache(trace, prices),
        Policy::Pcoe => replay_pcoe(trace, prices, cfg),
        Policy::ReferenceOracle => replay_oracle(trace, prices, cfg),
        Policy::CacheEverything => replay_static(trace, prices, cfg, None),
        Policy::StaticTtl(ttl) => replay_static(trace, prices, cfg, Some(ttl)),
    }
}

fn mtok(tokens: u64) -> f64 {
    tokens as f64 / 1e6
}

fn replay_no_cache(trace: &[Event], prices: &PriceSheet) -> Report {
    let mut r = Report::default();
    for e in trace {
        r.request_cost += mtok(e.total_tokens) * prices.input_per_mtok;
        r.requests += 1;
    }
    r.total_cost = r.request_cost;
    r
}

/// Build a trie over the whole trace so baselines/oracle can identify the
/// deepest cacheable prefix node (>= `M_min`) on each request path. Read-only
/// lookups afterwards (`peek_path`) don't perturb it.
fn build_trie(trace: &[Event], prices: &PriceSheet, cfg: &Config) -> PrefixTrie {
    let mut trie = PrefixTrie::new(cfg.block_tokens, cfg.tau_decay_hours, cfg.max_nodes);
    for e in trace {
        trie.observe_path(&e.blocks, e.time_hours);
    }
    let _ = prices;
    trie
}

/// Deepest node on `path` that is worth caching for the baselines/oracle: it
/// meets the cacheable minimum **and** is genuinely shared (traversed by more
/// than one request). The shared-ness guard is what stops a naive baseline
/// from caching a unique per-request suffix — the engine avoids this via its
/// arrival-rate gate instead.
fn deepest_cacheable(trie: &PrefixTrie, prices: &PriceSheet, path: &[NodeId]) -> Option<NodeId> {
    path.iter().rev().copied().find(|&id| {
        trie.tokens(id) >= prices.min_cacheable_tokens && trie.node(id).stats.raw_count >= 2
    })
}

/// Cache-everything (`ttl = None`) or static-idle-TTL (`ttl = Some(h)`).
fn replay_static(trace: &[Event], prices: &PriceSheet, cfg: &Config, ttl: Option<f64>) -> Report {
    let trie = build_trie(trace, prices, cfg);
    let mut r = Report::default();
    // Per cached node: (opened_at, last_hit).
    let mut live: HashMap<NodeId, (f64, f64)> = HashMap::new();
    let end = trace.last().map_or(0.0, |e| e.time_hours);

    for e in trace {
        let now = e.time_hours;
        // Expire idle nodes (static TTL only).
        if let Some(ttl_h) = ttl {
            let expired: Vec<NodeId> = live
                .iter()
                .filter(|(_, &(_, last))| now - last > ttl_h)
                .map(|(&id, _)| id)
                .collect();
            for id in expired {
                let (opened, last) = live.remove(&id).unwrap();
                // Storage accrues until the entry lapsed (last_hit + ttl).
                let closed = last + ttl_h;
                r.storage_cost +=
                    (closed - opened).max(0.0) * mtok(trie.tokens(id)) * storage(prices);
                r.token_hours += (closed - opened).max(0.0) * mtok(trie.tokens(id));
                r.deletes += 1;
            }
        }

        let path = trie.peek_path(&e.blocks);
        let node = deepest_cacheable(&trie, prices, &path);
        let mut cached_tokens = 0u64;
        if let Some(id) = node {
            match live.get_mut(&id) {
                Some((_, last)) => {
                    *last = now;
                    cached_tokens = trie.tokens(id);
                }
                None => {
                    // Create on first sight (or after expiry).
                    r.create_cost += mtok(trie.tokens(id)) * prices.input_per_mtok;
                    r.creates += 1;
                    live.insert(id, (now, now));
                    // The creating request itself reads uncached (create is for
                    // future requests), matching the engine's semantics.
                }
            }
        }
        r.request_cost += mtok(cached_tokens) * prices.cached_read_per_mtok
            + mtok(e.total_tokens - cached_tokens) * prices.input_per_mtok;
        if cached_tokens > 0 {
            r.cached_requests += 1;
        }
        r.requests += 1;
    }

    // Close everything still live at trace end.
    for (id, (opened, last)) in live {
        let closed = match ttl {
            Some(ttl_h) => (last + ttl_h).min(end).max(opened),
            None => end,
        };
        r.storage_cost += (closed - opened).max(0.0) * mtok(trie.tokens(id)) * storage(prices);
        r.token_hours += (closed - opened).max(0.0) * mtok(trie.tokens(id));
    }

    r.total_cost = r.create_cost + r.storage_cost + r.request_cost;
    r
}

fn storage(prices: &PriceSheet) -> f64 {
    prices.storage_per_mtok_hour.unwrap_or(0.0)
}

/// Replay through the real PCOE engine with an instantly-successful adapter.
fn replay_pcoe(trace: &[Event], prices: &PriceSheet, cfg: &Config) -> Report {
    let mut eng = Engine::new(prices.clone(), cfg.clone());
    let mut r = Report::default();
    // Open storage intervals per node: (opened_at, tokens, expires_at). We track
    // the provider-side expiry ourselves so that a cache the engine lets lapse
    // *silently* (its TTL runs out mid-hold without an explicit ski-rental
    // Delete — nothing to delete remotely) is still billed to its true expiry,
    // not left open. Relying on Delete actions alone would leak such intervals.
    let mut open: HashMap<NodeId, (f64, u64, f64)> = HashMap::new();
    let tick_dt = cfg.extend_lead_hours.max(1.0 / 60.0);
    let tau_hold = prices.tau_hold_hours().unwrap_or(0.0);

    let bill = |r: &mut Report, t0: f64, tokens: u64, closed: f64| {
        let held = (closed - t0).max(0.0);
        r.storage_cost += held * mtok(tokens) * storage(prices);
        r.token_hours += held * mtok(tokens);
    };

    // Close any interval whose provider TTL has lapsed by `clock`, billing to
    // the exact expiry — this is the silent provider-side auto-expiry.
    let sweep = |r: &mut Report, open: &mut HashMap<NodeId, (f64, u64, f64)>, clock: f64| {
        let lapsed: Vec<NodeId> = open
            .iter()
            .filter(|(_, &(_, _, exp))| exp <= clock)
            .map(|(&id, _)| id)
            .collect();
        for id in lapsed {
            let (t0, tokens, exp) = open.remove(&id).unwrap();
            bill(r, t0, tokens, exp);
        }
    };

    let apply = |r: &mut Report,
                 open: &mut HashMap<NodeId, (f64, u64, f64)>,
                 a: &Action,
                 now: f64| match *a {
        // A request-driven create and a predicted pre-create have the same cost
        // shape (write once at input rate, then storage runs). The pre-create's
        // payoff is that the next request reads cached instead of paying the
        // cold-start miss — it shows up as a lower request cost, not here.
        Action::Create { node, tokens, .. } | Action::PreCreate { node, tokens, .. } => {
            r.create_cost += mtok(tokens) * prices.input_per_mtok;
            r.creates += 1;
            open.insert(node, (now, tokens, now + tau_hold));
        }
        Action::Extend {
            node,
            expires_at_hours,
        } => {
            r.extends += 1;
            if let Some(entry) = open.get_mut(&node) {
                entry.2 = expires_at_hours;
            }
        }
        Action::Delete { node } => {
            r.deletes += 1;
            if let Some((t0, tokens, _)) = open.remove(&node) {
                bill(r, t0, tokens, now);
            }
        }
    };

    let mut next_tick = trace.first().map_or(0.0, |e| e.time_hours);
    for e in trace {
        let now = e.time_hours;
        // Drive maintenance ticks up to this event.
        while next_tick < now {
            for a in eng.tick(next_tick) {
                // A tick can emit a predictive PreCreate; confirm it like the
                // instantly-successful adapter does for request-driven creates.
                if let Action::PreCreate { node, .. } = a {
                    eng.confirm_create(node, next_tick);
                }
                apply(&mut r, &mut open, &a, next_tick);
            }
            sweep(&mut r, &mut open, next_tick);
            next_tick += tick_dt;
        }
        let obs = eng.observe(&e.blocks, now);
        for a in &obs.actions {
            if let Action::Create { node, .. } = a {
                eng.confirm_create(*node, now); // optimistic adapter
            }
            apply(&mut r, &mut open, a, now);
        }
        let ct = obs.cached_tokens.min(e.total_tokens);
        r.request_cost += mtok(ct) * prices.cached_read_per_mtok
            + mtok(e.total_tokens - ct) * prices.input_per_mtok;
        if ct > 0 {
            r.cached_requests += 1;
        }
        r.requests += 1;
    }

    // Drain: keep ticking past the last event until every cache is closed, so
    // end-of-trace storage is billed to the ski-rental exit, not the horizon.
    // Only *close out* existing caches here — a predictive pre-create fired
    // after the last event serves no request (there is none left in the trace),
    // so opening new intervals in the drain would bill speculative storage that
    // never happens in reality. In production every predicted return has real
    // traffic behind it; the drain is a pure end-of-trace artifact.
    let mut guard = 0;
    while !open.is_empty() && guard < 100_000 {
        for a in eng.tick(next_tick) {
            match a {
                Action::Create { .. } | Action::PreCreate { .. } => {}
                _ => apply(&mut r, &mut open, &a, next_tick),
            }
        }
        sweep(&mut r, &mut open, next_tick);
        next_tick += tick_dt;
        guard += 1;
    }
    // Anything still open (shouldn't happen) is closed at the final tick.
    for (_, (t0, tokens, _)) in open {
        bill(&mut r, t0, tokens, next_tick);
    }

    r.total_cost = r.create_cost + r.storage_cost + r.request_cost;
    r
}

/// Per-family offline-optimal ski-rental (perfect hindsight).
///
/// Groups requests by their deepest cacheable prefix node and runs the O(n)
/// ski-rental DP (DESIGN.md §3.3 offline form) on each group's timestamps.
/// Requests with no cacheable prefix pay full input. Suffix cost is added
/// uniformly so totals compare like-for-like.
fn replay_oracle(trace: &[Event], prices: &PriceSheet, cfg: &Config) -> Report {
    let trie = build_trie(trace, prices, cfg);
    let mut r = Report {
        requests: trace.len() as u64,
        ..Default::default()
    };

    // Group event indices by deepest cacheable node.
    let mut groups: HashMap<NodeId, Vec<usize>> = HashMap::new();
    for (i, e) in trace.iter().enumerate() {
        let path = trie.peek_path(&e.blocks);
        match deepest_cacheable(&trie, prices, &path) {
            Some(id) => groups.entry(id).or_default().push(i),
            None => {
                // No cacheable prefix — always full input.
                r.request_cost += mtok(e.total_tokens) * prices.input_per_mtok;
            }
        }
    }

    let p_in = prices.input_per_mtok;
    let p_read = prices.cached_read_per_mtok;
    let p_store = storage(prices);

    for (node, idxs) in groups {
        let t_tokens = trie.tokens(node);
        let read = mtok(t_tokens) * p_read;
        let input = mtok(t_tokens) * p_in;
        let create = mtok(t_tokens) * p_in;
        let store_rate = mtok(t_tokens) * p_store;
        let times: Vec<f64> = idxs.iter().map(|&i| trace[i].time_hours).collect();

        // Suffix cost (non-cacheable tail) is paid on every request either way.
        for &i in &idxs {
            let suffix = trace[i].total_tokens.saturating_sub(t_tokens);
            r.request_cost += mtok(suffix) * p_in;
        }

        // O(n) ski-rental DP over prefix cost only.
        // dp[i] = min(dp[i-1] + input,  const_i + min_{j<=i} h(j))
        // h(j) = dp[j-1] - read*j - store_rate*t[j]   (dp[-1] = 0)
        let mut dp_prev = 0.0f64;
        let mut run_min = f64::INFINITY;
        let mut prefix_cost = 0.0;
        for (i, &t_i) in times.iter().enumerate() {
            let dp_jm1 = if i == 0 { 0.0 } else { dp_prev };
            let h_i = dp_jm1 - read * i as f64 - store_rate * t_i;
            run_min = run_min.min(h_i);
            let const_i = create + read * (i as f64 + 1.0) + store_rate * t_i;
            let cached_end = const_i + run_min;
            let uncached = dp_jm1 + input;
            let dp_i = cached_end.min(uncached);
            prefix_cost = dp_i;
            dp_prev = dp_i;
        }
        r.request_cost += prefix_cost;
        // The oracle's op counts / hit rate aren't meaningfully defined per the
        // DP (it's a cost lower-bound), so we leave those fields at their
        // request-cost contribution and don't fabricate creates/reads.
    }

    r.total_cost = r.request_cost;
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace;

    fn gemini() -> PriceSheet {
        PriceSheet::gemini_pro_like()
    }

    #[test]
    fn policies_order_as_expected_over_a_week() {
        // 200K-token prefix, 8h business day, silence overnight, several days.
        // PCOE's win requires multiple periods: a single day ending at close
        // has no overnight idle *in the trace* for cache-everything to leak
        // into, so the gap only opens once nights are present.
        let trace = trace::business_hours(782, 12.0, 9.0, 17.0, 5, 256, 99);
        let cfg = Config::default();
        let cmp = compare(&trace, &gemini(), &cfg);

        let no = cmp.get(Policy::NoCache).unwrap().total_cost;
        let every = cmp.get(Policy::CacheEverything).unwrap().total_cost;
        let pcoe = cmp.get(Policy::Pcoe).unwrap().total_cost;
        let oracle = cmp.get(Policy::ReferenceOracle).unwrap().total_cost;

        // PCOE beats naive caching, which beats no caching.
        assert!(pcoe < every, "pcoe {pcoe} !< cache-everything {every}");
        assert!(every < no, "cache-everything {every} !< no-cache {no}");
        // PCOE saves a large fraction versus no caching.
        assert!(
            cmp.get(Policy::Pcoe)
                .unwrap()
                .savings_vs(cmp.get(Policy::NoCache).unwrap())
                > 60.0
        );
        // And it's near the offline oracle.
        let ratio = cmp.competitive_ratio().unwrap();
        assert!((0.95..1.6).contains(&ratio), "competitive ratio {ratio}");
        assert!(
            oracle <= pcoe + 1e-9,
            "oracle {oracle} should be <= pcoe {pcoe}"
        );
    }

    #[test]
    fn oracle_dp_matches_hand_computation_single_hit() {
        // One request, single cacheable prefix: never cache (create+read > input).
        let prices = gemini();
        let t_tokens = 5120u64;
        let ev = vec![Event {
            time_hours: 0.0,
            total_tokens: t_tokens,
            blocks: (0..20u64).collect(),
        }];
        let cfg = Config::default();
        let r = replay(&ev, &prices, &cfg, Policy::ReferenceOracle);
        let expected = mtok(t_tokens) * prices.input_per_mtok;
        assert!((r.total_cost - expected).abs() < 1e-12, "{}", r.total_cost);
    }

    #[test]
    fn oracle_caches_a_tight_cluster() {
        // Many requests 1 minute apart: the oracle caches once (create + reads +
        // tiny storage) rather than paying full input each time.
        let prices = gemini();
        let t_tokens = 5120u64;
        let blocks: Vec<u64> = (0..20u64).collect();
        let ev: Vec<Event> = (0..30)
            .map(|i| Event {
                time_hours: i as f64 / 60.0,
                total_tokens: t_tokens,
                blocks: blocks.clone(),
            })
            .collect();
        let cfg = Config::default();
        let r = replay(&ev, &prices, &cfg, Policy::ReferenceOracle);
        let no_cache = 30.0 * mtok(t_tokens) * prices.input_per_mtok;
        // Cached: create + 30 reads + ~29min storage.
        let cached_est = mtok(t_tokens) * prices.input_per_mtok
            + 30.0 * mtok(t_tokens) * prices.cached_read_per_mtok
            + (29.0 / 60.0) * mtok(t_tokens) * prices.storage_per_mtok_hour.unwrap();
        assert!((r.total_cost - cached_est).abs() < 1e-9, "{}", r.total_cost);
        assert!(r.total_cost < no_cache);
    }

    #[test]
    fn static_ttl_and_pcoe_beat_cache_everything_overnight() {
        // A prefix hit during the day then idle overnight: both a fixed idle-TTL
        // and PCOE stop paying storage after the last hit; cache-everything
        // doesn't. Needs multiple days so the overnight idle is in the trace.
        let trace = trace::business_hours(782, 12.0, 9.0, 17.0, 7, 256, 3);
        let cfg = Config::default();
        let every = replay(&trace, &gemini(), &cfg, Policy::CacheEverything);
        let static_1h = replay(&trace, &gemini(), &cfg, Policy::StaticTtl(1.0));
        let pcoe = replay(&trace, &gemini(), &cfg, Policy::Pcoe);
        assert!(static_1h.total_cost < every.total_cost);
        assert!(pcoe.total_cost < every.total_cost);
        // On a clean single-prefix metronome workload a *well-chosen* fixed TTL
        // is hard to beat: it deletes promptly after the day and re-creates each
        // morning, which is nearly optimal when the idle pattern never varies.
        // PCOE lands ~5-6% above it — the online ski-rental tax (the tau_hold
        // hold after each last hit, plus the morning rate re-warmup) against a
        // hindsight-perfect hand-tune. PCOE's value is adapting when no single
        // TTL fits (multi-tenant, bursty), not winning this metronome; assert it
        // stays in the same ballpark, not that it strictly wins.
        assert!(
            pcoe.total_cost < static_1h.total_cost * 1.07,
            "pcoe {} should be within 7% of static-1h {}",
            pcoe.total_cost,
            static_1h.total_cost
        );
    }

    #[test]
    fn bursty_learned_hold_improves_but_headroom_remains() {
        // Bimodal bursty traffic (hot phases separated by cold lulls whose gaps
        // straddle tau_hold) is PCOE's hardest case: a greedy per-gap
        // ski-rental would delete during a lull right before the next burst.
        // The learned per-node hold recovers most of that — a prefix that keeps
        // getting hit right after a delete grows its hold and rides through the
        // lulls. Averaged over seeds the competitive ratio is well below the
        // ~1.29 a purely greedy controller gives, but real headroom remains
        // (the hot→cold transitions and the growth ramp can't be won online —
        // that needs burst prediction, DESIGN.md §8). Averaged over seeds so
        // the assertion doesn't ride on one worst-case instance.
        let cfg = Config::default();
        let ratios: Vec<f64> = (1..=8u64)
            .map(|seed| {
                let trace = trace::poisson_bursty(390, 60.0, 2.0, 1.0, 168.0, 256, seed);
                let cmp = compare(&trace, &gemini(), &cfg);
                let pcoe = cmp.get(Policy::Pcoe).unwrap();
                let no = cmp.get(Policy::NoCache).unwrap();
                assert!(pcoe.total_cost < no.total_cost); // caching always beats not
                pcoe.total_cost / cmp.get(Policy::ReferenceOracle).unwrap().total_cost
            })
            .collect();
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        assert!(
            mean < 1.28,
            "learned-hold should pull the mean ratio below the greedy ~1.29: {mean}"
        );
        assert!(
            mean > 1.05,
            "but bursty headroom remains (needs burst prediction): {mean}"
        );
    }

    #[test]
    fn multi_tenant_fleet_pcoe_wins_big() {
        // 12 tenants, staggered around the clock: no human can hand-tune this.
        let trace = trace::multi_tenant(12, 400, 15.0, 8.0, 1, 256, 5);
        let cfg = Config::default();
        let cmp = compare(&trace, &gemini(), &cfg);
        let pcoe = cmp.get(Policy::Pcoe).unwrap();
        let no = cmp.get(Policy::NoCache).unwrap();
        let every = cmp.get(Policy::CacheEverything).unwrap();
        assert!(
            pcoe.savings_vs(no) > 50.0,
            "savings {}",
            pcoe.savings_vs(no)
        );
        assert!(pcoe.total_cost < every.total_cost);
        assert!(pcoe.hit_rate() > 0.5, "hit rate {}", pcoe.hit_rate());
    }
}
