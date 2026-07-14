//! Decayed prefix trie over hashed token blocks — PCOE's traffic model
//! (DESIGN.md §3.1).
//!
//! Each outgoing request is chunked into fixed-size token blocks and each block
//! hashed; the hash sequence is inserted as a path. Nodes hold only hashes,
//! decayed counters, and timestamps — **no prompt content** — so the structure is
//! privacy-safe to keep resident or persist.

use std::collections::HashMap;

/// Index of a node in the trie arena.
pub type NodeId = usize;

/// Gap-histogram bucket edges in hours: `< 5 min`, `5 min .. 1 h`, `> 1 h`.
/// Deliberately aligned with the common provider TTL tiers so
/// [`NodeStats::gap_fraction_gt`] is exact at tier boundaries.
pub const GAP_EDGES_HOURS: [f64; 2] = [5.0 / 60.0, 1.0];

/// Per-node decayed traffic statistics.
#[derive(Clone, Debug)]
pub struct NodeStats {
    /// EWMA arrival-rate estimate, requests/hour, valued as of `t_last`.
    pub lambda: f64,
    /// Time of the last traversal, hours.
    pub t_last: f64,
    /// Decayed traversal count (confidence proxy).
    pub samples: f64,
    /// Raw (undecayed) traversal count — how many requests have ever passed
    /// through this prefix. Distinguishes a shared prefix (count ≫ 1) from a
    /// unique per-request suffix (count == 1).
    pub raw_count: u64,
    /// Decayed inter-arrival gap histogram (see [`GAP_EDGES_HOURS`]).
    pub gap_mass: [f64; 3],
}

impl NodeStats {
    fn new() -> Self {
        NodeStats {
            lambda: 0.0,
            t_last: 0.0,
            samples: 0.0,
            raw_count: 0,
            gap_mass: [0.0; 3],
        }
    }

    /// Record a traversal at `now` with decay constant `tau` (hours). Decay is
    /// applied lazily — idle nodes cost nothing between traversals.
    fn observe(&mut self, now: f64, tau: f64) {
        self.raw_count = self.raw_count.saturating_add(1);
        if self.samples == 0.0 {
            self.lambda = 1.0 / tau;
            self.samples = 1.0;
            self.t_last = now;
            return;
        }
        let dt = (now - self.t_last).max(0.0);
        let decay = (-dt / tau).exp();
        self.lambda = self.lambda * decay + 1.0 / tau;
        for m in self.gap_mass.iter_mut() {
            *m *= decay;
        }
        let bucket = if dt < GAP_EDGES_HOURS[0] {
            0
        } else if dt < GAP_EDGES_HOURS[1] {
            1
        } else {
            2
        };
        self.gap_mass[bucket] += 1.0;
        self.samples = self.samples * decay + 1.0;
        self.t_last = now;
    }

    /// The rate estimate decayed to `now` (read-only; does not mutate).
    pub fn decayed_lambda(&self, now: f64, tau: f64) -> f64 {
        if self.samples == 0.0 {
            return 0.0;
        }
        self.lambda * (-((now - self.t_last).max(0.0)) / tau).exp()
    }

    /// Fraction of observed (decayed) gap mass *guaranteed* to be above
    /// `hours`: only buckets whose lower edge is at least `hours` count, so the
    /// estimate is conservative (never overstates) at bucket resolution, and
    /// exact when `hours` equals a bucket edge — which the default edges are
    /// chosen to do for the 5-minute / 1-hour provider tiers.
    pub fn gap_fraction_gt(&self, hours: f64) -> f64 {
        let total: f64 = self.gap_mass.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        let mut above = 0.0;
        if hours <= 0.0 {
            above += self.gap_mass[0];
        }
        if GAP_EDGES_HOURS[0] >= hours {
            above += self.gap_mass[1];
        }
        if GAP_EDGES_HOURS[1] >= hours {
            above += self.gap_mass[2];
        }
        above / total
    }
}

/// Cache state of a trie node, mirroring the provider-side object PCOE manages.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CacheState {
    /// No provider-side cache for this prefix.
    Uncached,
    /// A create was requested but not yet confirmed by the adapter
    /// ([`crate::Engine::confirm_create`]). Micro-batching defers followers
    /// while a node is pending (DESIGN.md §3.4b).
    Pending {
        /// When the create action was emitted, hours.
        since_hours: f64,
    },
    /// A provider-side cache is (believed) live for this prefix.
    Cached {
        /// When the provider-side entry expires unless extended, hours.
        expires_at_hours: f64,
        /// TTL tier the entry was bought at, hours.
        ttl_hours: f64,
    },
}

/// One trie node: an edge hash from its parent plus decayed statistics.
#[derive(Clone, Debug)]
pub struct Node {
    /// Block hash on the edge from `parent` to this node.
    pub edge: u64,
    /// Parent node (`None` only for the root).
    pub parent: Option<NodeId>,
    /// Children keyed by edge block hash.
    pub children: HashMap<u64, NodeId>,
    /// Depth in blocks (root = 0).
    pub depth_blocks: u64,
    /// Decayed traffic statistics.
    pub stats: NodeStats,
    /// Provider cache state for the prefix ending at this node.
    pub state: CacheState,
    /// Last time `state` flipped (for hysteresis dwell).
    pub last_flip: f64,
    /// Learned idle-hold time for this prefix, hours (0 = use the default
    /// `tau_hold`). Grown when a delete turns out to have been premature —
    /// traffic returned right after we let the cache go — and decayed back
    /// toward the default otherwise. Lets a prefix with recurring lulls hold
    /// through them without changing the common case (DESIGN.md §3.3).
    pub hold_hint: f64,
    /// Time the last cache for this prefix was deleted, hours
    /// (`NEG_INFINITY` if never). Used to detect a premature delete.
    pub last_delete: f64,
    alive: bool,
}

/// The decayed prefix trie (arena-allocated).
pub struct PrefixTrie {
    nodes: Vec<Node>,
    free: Vec<NodeId>,
    alive_count: usize,
    /// Tokens per block used when this trie was built. Callers must chunk with
    /// the same block size (see [`crate::chunk`]).
    pub block_tokens: u64,
    /// EWMA decay constant, hours.
    pub tau_decay: f64,
    max_nodes: usize,
}

impl PrefixTrie {
    /// The root node id (empty prefix).
    pub const ROOT: NodeId = 0;

    /// Create an empty trie. `max_nodes` bounds memory: once reached, new
    /// (previously unseen) suffix nodes are simply not materialized.
    pub fn new(block_tokens: u64, tau_decay: f64, max_nodes: usize) -> Self {
        let root = Node {
            edge: 0,
            parent: None,
            children: HashMap::new(),
            depth_blocks: 0,
            stats: NodeStats::new(),
            state: CacheState::Uncached,
            last_flip: f64::NEG_INFINITY,
            hold_hint: 0.0,
            last_delete: f64::NEG_INFINITY,
            alive: true,
        };
        PrefixTrie {
            nodes: vec![root],
            free: Vec::new(),
            alive_count: 1,
            block_tokens,
            tau_decay,
            max_nodes,
        }
    }

    /// Number of live nodes (including the root).
    pub fn len(&self) -> usize {
        self.alive_count
    }

    /// True when only the root exists.
    pub fn is_empty(&self) -> bool {
        self.alive_count <= 1
    }

    /// Insert/traverse `blocks` as a path from the root at time `now`, updating
    /// each traversed node's statistics. Returns the node ids along the path
    /// (may be shorter than `blocks` if the node budget is exhausted).
    pub fn observe_path(&mut self, blocks: &[u64], now: f64) -> Vec<NodeId> {
        let mut path = Vec::with_capacity(blocks.len());
        let mut cur = Self::ROOT;
        for &b in blocks {
            let next = match self.nodes[cur].children.get(&b) {
                Some(&id) => id,
                None => {
                    if self.alive_count >= self.max_nodes {
                        break;
                    }
                    let depth = self.nodes[cur].depth_blocks + 1;
                    let node = Node {
                        edge: b,
                        parent: Some(cur),
                        children: HashMap::new(),
                        depth_blocks: depth,
                        stats: NodeStats::new(),
                        state: CacheState::Uncached,
                        last_flip: f64::NEG_INFINITY,
                        hold_hint: 0.0,
                        last_delete: f64::NEG_INFINITY,
                        alive: true,
                    };
                    let id = match self.free.pop() {
                        Some(slot) => {
                            self.nodes[slot] = node;
                            slot
                        }
                        None => {
                            self.nodes.push(node);
                            self.nodes.len() - 1
                        }
                    };
                    self.alive_count += 1;
                    self.nodes[cur].children.insert(b, id);
                    id
                }
            };
            self.nodes[next].stats.observe(now, self.tau_decay);
            path.push(next);
            cur = next;
        }
        path
    }

    /// Read-only path lookup: walk `blocks` from the root without recording a
    /// traversal or materializing nodes. Stops at the first unseen block.
    /// Used for cost quotes (e.g. cross-provider routing) that must not
    /// pollute the traffic statistics.
    pub fn peek_path(&self, blocks: &[u64]) -> Vec<NodeId> {
        let mut path = Vec::new();
        let mut cur = Self::ROOT;
        for b in blocks {
            match self.nodes[cur].children.get(b) {
                Some(&id) => {
                    path.push(id);
                    cur = id;
                }
                None => break,
            }
        }
        path
    }

    /// Prefix length in tokens at `id`.
    pub fn tokens(&self, id: NodeId) -> u64 {
        self.nodes[id].depth_blocks * self.block_tokens
    }

    /// Immutable node access.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    /// Mutable node access.
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id]
    }

    /// Child ids of `id`, sorted by edge hash for determinism.
    pub fn children_sorted(&self, id: NodeId) -> Vec<NodeId> {
        let mut edges: Vec<(u64, NodeId)> = self.nodes[id]
            .children
            .iter()
            .map(|(&e, &c)| (e, c))
            .collect();
        edges.sort_unstable_by_key(|&(e, _)| e);
        edges.into_iter().map(|(_, c)| c).collect()
    }

    /// Garbage-collect cold leaves: uncached leaf nodes idle for at least
    /// `min_idle_hours` whose decayed rate fell below `min_lambda`. Runs to a
    /// fixpoint so cold chains unwind bottom-up. Returns nodes removed.
    pub fn gc(&mut self, now: f64, min_lambda: f64, min_idle_hours: f64) -> usize {
        let mut removed = 0;
        loop {
            let victims: Vec<NodeId> = (1..self.nodes.len())
                .filter(|&id| {
                    let n = &self.nodes[id];
                    n.alive
                        && n.children.is_empty()
                        && n.state == CacheState::Uncached
                        && now - n.stats.t_last >= min_idle_hours
                        && n.stats.decayed_lambda(now, self.tau_decay) < min_lambda
                })
                .collect();
            if victims.is_empty() {
                break;
            }
            for id in victims {
                let (parent, edge) = (self.nodes[id].parent, self.nodes[id].edge);
                if let Some(p) = parent {
                    self.nodes[p].children.remove(&edge);
                }
                self.nodes[id].alive = false;
                self.free.push(id);
                self.alive_count -= 1;
                removed += 1;
            }
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewma_rate_converges_toward_inverse_gap() {
        // Requests every 5 minutes with tau = 1h: lambda_1 = 1, lambda_2 = e^{-1/12} + 1 ...
        let mut s = NodeStats::new();
        let gap = 5.0 / 60.0;
        s.observe(0.0, 1.0);
        assert!((s.lambda - 1.0).abs() < 1e-12);
        s.observe(gap, 1.0);
        assert!((s.lambda - (1.0f64 * (-gap).exp() + 1.0)).abs() < 1e-9);
        for i in 2..200 {
            s.observe(i as f64 * gap, 1.0);
        }
        // Steady state (1/tau)/(1 - e^{-g/tau}) ~= 12.5 for g = 1/12.
        let steady = 1.0 / (1.0 - (-gap).exp());
        assert!((s.lambda - steady).abs() / steady < 0.01);
    }

    #[test]
    fn gap_histogram_buckets_and_fractions() {
        let mut s = NodeStats::new();
        s.observe(0.0, 1000.0); // huge tau: negligible decay for this test
        s.observe(0.02, 1000.0); // ~1.2 min gap -> bucket 0
        s.observe(0.52, 1000.0); // 30 min gap  -> bucket 1
        s.observe(2.52, 1000.0); // 2 h gap     -> bucket 2
        assert!(s.gap_fraction_gt(5.0 / 60.0) > 0.6); // 2 of 3 gaps above 5 min
        assert!(s.gap_fraction_gt(1.0) > 0.3 && s.gap_fraction_gt(1.0) < 0.4);
    }

    #[test]
    fn paths_share_prefix_nodes() {
        let mut t = PrefixTrie::new(256, 1.0, 1000);
        let a = t.observe_path(&[1, 2, 3], 0.0);
        let b = t.observe_path(&[1, 2, 9], 0.1);
        assert_eq!(a[0], b[0]);
        assert_eq!(a[1], b[1]);
        assert_ne!(a[2], b[2]);
        assert_eq!(t.tokens(a[1]), 512);
        // Shared node saw both traversals.
        assert!(t.node(a[1]).stats.samples > t.node(a[2]).stats.samples);
    }

    #[test]
    fn gc_unwinds_cold_chains_but_keeps_hot_and_cached() {
        let mut t = PrefixTrie::new(256, 1.0, 1000);
        let cold = t.observe_path(&[7, 8, 9], 0.0);
        let hot = t.observe_path(&[1, 2], 100.0);
        t.node_mut(hot[1]).state = CacheState::Cached {
            expires_at_hours: 200.0,
            ttl_hours: 1.0,
        };
        let removed = t.gc(100.0, 0.5, 24.0);
        assert_eq!(removed, cold.len());
        assert_eq!(t.len(), 1 + 2); // root + hot chain
    }

    #[test]
    fn node_budget_truncates_paths() {
        let mut t = PrefixTrie::new(256, 1.0, 3); // root + 2 nodes
        let p = t.observe_path(&[1, 2, 3, 4], 0.0);
        assert_eq!(p.len(), 2);
    }
}
