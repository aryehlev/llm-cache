//! Request shaping: prefix-stability boundary detection (DESIGN.md §3.4a).
//!
//! The trie itself is the volatility detector: a node with high fan-out —
//! many rarely-repeated children — marks the point where a request family's
//! stable shared content ends and per-request volatile content begins. The
//! boundary is where a cache point belongs, and everything the application can
//! reorder (timestamps, user IDs, per-request footers) should render *after*
//! it. PCOE cannot see content, so it reports the boundary and its value; the
//! application decides what to move.

use crate::plan::{cost_rate, savings_rate};
use crate::price::PriceSheet;
use crate::trie::{NodeId, PrefixTrie};

/// A detected stable→volatile boundary on a request path.
#[derive(Clone, Copy, Debug)]
pub struct ShapeHint {
    /// Deepest node on the path where the traffic fans out — the last block of
    /// stable shared content.
    pub boundary_node: NodeId,
    /// Stable prefix length in tokens. Content after this offset varies per
    /// request; keep volatile fields after it (or move them there).
    pub stable_tokens: u64,
    /// Number of distinct continuations observed after the boundary.
    pub fanout: usize,
    /// Net savings rate ($/hour) a cache point at the boundary would earn at
    /// current traffic — the value of keeping (or making) this prefix stable.
    pub net_savings_per_hour: f64,
}

/// Find the stable→volatile boundary on `path`: the deepest node whose
/// fan-out is at least `min_fanout`. Returns `None` when the path never fans
/// out (a unique or single-family request — nothing to shape yet).
///
/// `min_fanout` trades sensitivity for confidence; 3 is a reasonable default
/// (two distinct continuations can just be an A/B pair, three start looking
/// like a pattern).
pub fn stable_boundary(
    trie: &PrefixTrie,
    prices: &PriceSheet,
    path: &[NodeId],
    now: f64,
    min_fanout: usize,
) -> Option<ShapeHint> {
    for &id in path.iter().rev() {
        let node = trie.node(id);
        let fanout = node.children.len();
        if fanout >= min_fanout {
            let tokens = trie.tokens(id);
            let lambda = node.stats.decayed_lambda(now, trie.tau_decay);
            let s = savings_rate(prices, tokens, lambda);
            let (c, _) = cost_rate(prices, tokens, &node.stats, now, trie.tau_decay);
            return Some(ShapeHint {
                boundary_node: id,
                stable_tokens: tokens,
                fanout,
                net_savings_per_hour: s - c,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::price::PriceSheet;
    use crate::trie::PrefixTrie;

    #[test]
    fn boundary_lands_where_traffic_fans_out() {
        let mut t = PrefixTrie::new(256, 1.0, 100_000);
        let prefix: Vec<u64> = (0..20).collect(); // 5120 stable tokens
        let mut path = Vec::new();
        for i in 0..6u64 {
            let mut blocks = prefix.clone();
            blocks.push(0xAAAA + i); // unique volatile continuation
            path = t.observe_path(&blocks, i as f64 * (5.0 / 60.0));
        }
        let prices = PriceSheet::gemini_pro_like();
        let hint = stable_boundary(&t, &prices, &path, 0.5, 3).expect("boundary must be found");
        assert_eq!(hint.stable_tokens, 5120);
        assert_eq!(hint.fanout, 6);
        assert!(
            hint.net_savings_per_hour > 0.0,
            "hot stable prefix must be worth caching: {}",
            hint.net_savings_per_hour
        );
    }

    #[test]
    fn no_boundary_without_fanout() {
        let mut t = PrefixTrie::new(256, 1.0, 100_000);
        let blocks: Vec<u64> = (0..20).collect();
        let mut path = Vec::new();
        for i in 0..6 {
            path = t.observe_path(&blocks, i as f64 * (5.0 / 60.0));
        }
        let prices = PriceSheet::gemini_pro_like();
        assert!(stable_boundary(&t, &prices, &path, 0.5, 3).is_none());
    }
}
