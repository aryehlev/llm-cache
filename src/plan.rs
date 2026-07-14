//! Value function and breakpoint placement — the optimization core
//! (DESIGN.md §3.2).
//!
//! Selects the set of at most `K_max` cache points in the trie maximizing net
//! savings under *exclusive coverage* semantics: a request reads through its
//! deepest cached ancestor, so a shallower cache point only earns savings on
//! traffic that does **not** pass through a deeper cached one. This is a tree
//! knapsack, solved by DP over the trie.

use crate::price::{PriceSheet, Regime};
use crate::trie::{CacheState, NodeId, NodeStats, PrefixTrie};

/// A recommended cache point.
#[derive(Clone, Debug)]
pub struct Placement {
    /// Trie node at whose prefix boundary the cache point goes.
    pub node: NodeId,
    /// Prefix length in tokens covered by this cache point.
    pub tokens: u64,
    /// Chosen TTL tier in hours (write-premium regime); `None` on the storage
    /// regime, where lifetime is managed by the lifecycle controller instead.
    pub ttl_hours: Option<f64>,
    /// Expected net savings rate of this placement, $/hour (savings minus
    /// holding cost, at current traffic estimates).
    pub net_per_hour: f64,
}

/// Expected gross savings rate ($/hour) of serving `lambda` req/hour of
/// `tokens`-long prefixes from cache instead of paying full input price.
pub(crate) fn savings_rate(prices: &PriceSheet, tokens: u64, lambda: f64) -> f64 {
    lambda * (tokens as f64 / 1e6) * (prices.input_per_mtok - prices.cached_read_per_mtok)
}

/// Expected holding-cost rate ($/hour) of keeping this prefix cached, and the
/// chosen TTL tier where applicable.
///
/// - Storage regime: `T * p_store` — the meter runs regardless of traffic.
/// - Write-premium regime: expected cold-rewrite rate (gaps longer than the
///   tier TTL, estimated from the node's gap histogram) times the write
///   premium; the cheapest tier is selected (DESIGN.md §3.2, TTL-tier rule).
pub(crate) fn cost_rate(
    prices: &PriceSheet,
    tokens: u64,
    stats: &NodeStats,
    now: f64,
    tau: f64,
) -> (f64, Option<f64>) {
    let mtok = tokens as f64 / 1e6;
    if let Some(s) = prices.storage_per_mtok_hour {
        return (mtok * s, None);
    }
    let lambda = stats.decayed_lambda(now, tau);
    let mut best: Option<(f64, f64)> = None;
    for tier in &prices.write_tiers {
        let rewrites_per_hour = lambda * stats.gap_fraction_gt(tier.ttl_hours);
        let c = rewrites_per_hour * mtok * (tier.write_multiplier - 1.0) * prices.input_per_mtok;
        if best.map_or(true, |(bc, _)| c < bc) {
            best = Some((c, tier.ttl_hours));
        }
    }
    match best {
        Some((c, ttl)) => (c, Some(ttl)),
        None => (0.0, None),
    }
}

#[derive(Clone)]
struct DpEntry {
    /// Total net savings rate of the chosen placements, $/hour.
    value: f64,
    /// Sum of arrival rates covered by the *highest* chosen placements in this
    /// subtree — the traffic an ancestor placement must exclude.
    covered: f64,
    picks: Vec<Placement>,
}

impl DpEntry {
    fn zero() -> Self {
        DpEntry {
            value: 0.0,
            covered: 0.0,
            picks: Vec::new(),
        }
    }
}

/// Solve the budgeted placement problem over the whole trie.
///
/// `theta_up` / `theta_down` are hysteresis thresholds: a currently-uncached
/// node is only proposed when its savings exceed `theta_up x` its holding cost;
/// a currently-cached node is kept down to `theta_down x` (DESIGN.md §3.2).
/// Reported `net_per_hour` is always the un-thresholded `S - C`.
pub fn plan_breakpoints(
    trie: &PrefixTrie,
    prices: &PriceSheet,
    now: f64,
    theta_up: f64,
    theta_down: f64,
) -> Vec<Placement> {
    let k = prices.max_breakpoints;
    if k == 0 || matches!(prices.regime(), Regime::ShapeOnly) {
        return Vec::new();
    }
    let entries = solve(trie, prices, PrefixTrie::ROOT, now, k, theta_up, theta_down);
    entries
        .into_iter()
        .max_by(|a, b| a.value.total_cmp(&b.value))
        .map(|e| e.picks)
        .unwrap_or_default()
}

/// Returns, for each budget `0..=k`, the best entry for the subtree at `id`.
fn solve(
    trie: &PrefixTrie,
    prices: &PriceSheet,
    id: NodeId,
    now: f64,
    k: usize,
    theta_up: f64,
    theta_down: f64,
) -> Vec<DpEntry> {
    // Combine children with a small knapsack merge.
    let mut acc: Vec<DpEntry> = (0..=k).map(|_| DpEntry::zero()).collect();
    for child in trie.children_sorted(id) {
        let ch = solve(trie, prices, child, now, k, theta_up, theta_down);
        let mut merged = acc.clone();
        for i in 0..=k {
            for (j, cj) in ch.iter().enumerate().take(k - i + 1) {
                let value = acc[i].value + cj.value;
                if value > merged[i + j].value {
                    let mut picks = acc[i].picks.clone();
                    picks.extend(cj.picks.iter().cloned());
                    merged[i + j] = DpEntry {
                        value,
                        covered: acc[i].covered + cj.covered,
                        picks,
                    };
                }
            }
        }
        acc = merged;
    }

    // Option: place a cache point at this node.
    if id != PrefixTrie::ROOT {
        let node = trie.node(id);
        let tokens = trie.tokens(id);
        if tokens >= prices.min_cacheable_tokens {
            let lambda = node.stats.decayed_lambda(now, trie.tau_decay);
            let (c, ttl) = cost_rate(prices, tokens, &node.stats, now, trie.tau_decay);
            let threshold = if matches!(node.state, CacheState::Cached { .. }) {
                theta_down
            } else {
                theta_up
            };
            for kk in (1..=k).rev() {
                let base = &acc[kk - 1];
                let lambda_excl = (lambda - base.covered).max(0.0);
                let s = savings_rate(prices, tokens, lambda_excl);
                let net = s - c;
                if s > threshold * c && net > 0.0 {
                    let value = base.value + net;
                    if value > acc[kk].value {
                        let mut picks = base.picks.clone();
                        picks.push(Placement {
                            node: id,
                            tokens,
                            ttl_hours: ttl,
                            net_per_hour: net,
                        });
                        acc[kk] = DpEntry {
                            value,
                            covered: lambda,
                            picks,
                        };
                    }
                }
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::price::PriceSheet;
    use crate::trie::PrefixTrie;

    /// Build a trie with a shared 8-block prefix A, a hot 16-block branch A+B,
    /// and a cold branch A+C.
    fn hot_cold_trie() -> (PrefixTrie, Vec<NodeId>, Vec<NodeId>, f64) {
        let mut t = PrefixTrie::new(256, 1.0, 100_000);
        let a: Vec<u64> = (0..8).collect();
        let ab: Vec<u64> = a.iter().copied().chain(100..108).collect();
        let ac: Vec<u64> = a.iter().copied().chain(200..208).collect();
        let mut ab_path = Vec::new();
        for i in 0..30 {
            ab_path = t.observe_path(&ab, i as f64 * (2.0 / 60.0));
        }
        t.observe_path(&ac, 0.5);
        let ac_path = t.observe_path(&ac, 0.9);
        (t, ab_path, ac_path, 1.0)
    }

    #[test]
    fn dp_places_deepest_hot_node_and_picks_cheap_tier() {
        let (t, ab_path, _, now) = hot_cold_trie();
        let prices = PriceSheet::anthropic_sonnet_like();
        let picks = plan_breakpoints(&t, &prices, now, 1.2, 0.8);
        assert!(!picks.is_empty() && picks.len() <= prices.max_breakpoints);
        let deep = *ab_path.last().unwrap();
        let deep_pick = picks
            .iter()
            .find(|p| p.node == deep)
            .expect("deepest hot node must be placed");
        assert_eq!(deep_pick.tokens, 16 * 256);
        // All observed gaps are 2 minutes < 5 min, so no cold rewrites are
        // expected on the 5-minute tier: it must win over the 1-hour tier.
        assert_eq!(deep_pick.ttl_hours, Some(5.0 / 60.0));
        assert!(deep_pick.net_per_hour > 0.0);
    }

    #[test]
    fn exclusive_coverage_prevents_double_counting_along_a_chain() {
        let (t, ab_path, ac_path, now) = hot_cold_trie();
        let prices = PriceSheet::anthropic_sonnet_like();
        let picks = plan_breakpoints(&t, &prices, now, 1.2, 0.8);
        // Legitimate placements are branch endpoints (deep AB, deep AC) and the
        // fork point (end of A). Intermediate chain nodes carry the same
        // traffic as their deeper endpoint — zero exclusive rate — and must
        // never be selected.
        let legit = [
            *ab_path.last().unwrap(),
            *ac_path.last().unwrap(),
            ab_path[7],
        ];
        for p in &picks {
            assert!(
                legit.contains(&p.node),
                "unexpected placement at node {} ({} tokens)",
                p.node,
                p.tokens
            );
        }
    }

    #[test]
    fn tier_choice_flips_on_gap_histogram() {
        // A node whose gaps all fall between 5 minutes and 1 hour: the 5-minute
        // tier pays a rewrite premium on ~every arrival, the 1-hour tier never
        // does — the 1-hour tier must be chosen.
        let mut t = PrefixTrie::new(256, 1000.0, 1000);
        let blocks: Vec<u64> = (0..8).collect();
        let mut path = Vec::new();
        for i in 0..20 {
            path = t.observe_path(&blocks, i as f64 * 0.5); // 30-minute gaps
        }
        let prices = PriceSheet::anthropic_sonnet_like();
        let node = *path.last().unwrap();
        let (cost, ttl) = cost_rate(
            &prices,
            t.tokens(node),
            &t.node(node).stats,
            10.0,
            t.tau_decay,
        );
        assert_eq!(ttl, Some(1.0));
        assert!(cost < 1e-9); // no gaps above 1 hour -> no expected rewrites
    }

    #[test]
    fn storage_regime_respects_single_breakpoint_budget() {
        let (t, _, _, now) = hot_cold_trie();
        let prices = PriceSheet::gemini_pro_like(); // K_max = 1
        let picks = plan_breakpoints(&t, &prices, now, 1.2, 0.8);
        assert!(picks.len() <= 1);
    }
}
