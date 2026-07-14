//! Provider adapters: translate engine [`Action`]s and [`Observation`]s into
//! concrete provider API operations (DESIGN.md §5).
//!
//! The adapters are deliberately **wire-agnostic**: PCOE never retains prompt
//! content (privacy invariant, §3.5), so it cannot build API payloads itself.
//! Instead each adapter tells the application — which still holds the request
//! content — exactly *what* call to make with *which slice* of its own data.
//! Executing the HTTP call, and reporting back via [`crate::Engine::confirm_create`] /
//! [`crate::Engine::mark_failed`], stays with the application.

pub mod gemini {
    //! Gemini explicit caching (`cachedContents.*`), storage-metered regime.

    use std::collections::HashMap;

    use crate::engine::{Action, Observation};
    use crate::trie::NodeId;

    /// A concrete Gemini API operation for the application to execute.
    #[derive(Clone, Debug, PartialEq)]
    pub enum GeminiCall {
        /// `cachedContents.create`: build a `CachedContent` from the **first
        /// `prefix_tokens` tokens of the current request** (the application
        /// slices its own content), with the given TTL. On success, call
        /// [`GeminiAdapter::register`] with the returned resource name and
        /// [`crate::Engine::confirm_create`]; on failure,
        /// [`crate::Engine::mark_failed`].
        CreateCachedContent {
            /// Engine node this cache corresponds to.
            node: NodeId,
            /// How many leading tokens of the request to cache.
            prefix_tokens: u64,
            /// Requested TTL, seconds.
            ttl_seconds: u64,
        },
        /// `cachedContents.create` **ahead of a predicted return** — there is no
        /// in-flight request, so the application must build the `CachedContent`
        /// from its **retained stable-prefix content** (the fixed system prompt
        /// / corpus it re-sends every request), not from a request slice. Same
        /// success/failure callbacks as [`GeminiCall::CreateCachedContent`].
        PreCreateCachedContent {
            /// Engine node this cache corresponds to.
            node: NodeId,
            /// How many leading tokens of the stored prefix to cache.
            prefix_tokens: u64,
            /// Requested TTL, seconds.
            ttl_seconds: u64,
        },
        /// `cachedContents.patch`: push the entry's expiry out.
        UpdateExpiry {
            /// Engine node.
            node: NodeId,
            /// Provider resource name (e.g. `cachedContents/abc123`).
            cache_name: String,
            /// New TTL from now, seconds.
            ttl_seconds: u64,
        },
        /// `cachedContents.delete`.
        DeleteCachedContent {
            /// Engine node.
            node: NodeId,
            /// Provider resource name.
            cache_name: String,
        },
    }

    /// Stateful translation layer: engine actions → Gemini calls, plus the
    /// NodeId → `cachedContents/...` handle registry.
    #[derive(Default)]
    pub struct GeminiAdapter {
        handles: HashMap<NodeId, String>,
    }

    impl GeminiAdapter {
        /// New adapter with an empty handle registry.
        pub fn new() -> Self {
            Self::default()
        }

        /// Record the provider resource name returned by a successful create.
        /// Pair with [`crate::Engine::confirm_create`].
        pub fn register(&mut self, node: NodeId, cache_name: String) {
            self.handles.insert(node, cache_name);
        }

        /// The registered handle for a node, if any.
        pub fn handle(&self, node: NodeId) -> Option<&str> {
            self.handles.get(&node).map(String::as_str)
        }

        /// The `cachedContent` resource name this request should reference
        /// (from [`Observation::deepest_cached`]), if its create was
        /// registered.
        pub fn reference_for(&self, obs: &Observation) -> Option<&str> {
            obs.deepest_cached.and_then(|id| self.handle(id))
        }

        /// Translate engine actions into Gemini calls. Extends/deletes whose
        /// node has no registered handle are dropped (fail-open: the engine
        /// state machine already tolerates lost actions).
        pub fn translate(&mut self, actions: &[Action], now_hours: f64) -> Vec<GeminiCall> {
            let mut calls = Vec::new();
            for a in actions {
                match *a {
                    Action::Create {
                        node,
                        tokens,
                        ttl_hours,
                    } => calls.push(GeminiCall::CreateCachedContent {
                        node,
                        prefix_tokens: tokens,
                        ttl_seconds: (ttl_hours * 3600.0).ceil() as u64,
                    }),
                    Action::PreCreate {
                        node,
                        tokens,
                        ttl_hours,
                    } => calls.push(GeminiCall::PreCreateCachedContent {
                        node,
                        prefix_tokens: tokens,
                        ttl_seconds: (ttl_hours * 3600.0).ceil() as u64,
                    }),
                    Action::Extend {
                        node,
                        expires_at_hours,
                    } => {
                        if let Some(name) = self.handles.get(&node) {
                            let ttl = ((expires_at_hours - now_hours) * 3600.0).ceil();
                            calls.push(GeminiCall::UpdateExpiry {
                                node,
                                cache_name: name.clone(),
                                ttl_seconds: ttl.max(0.0) as u64,
                            });
                        }
                    }
                    Action::Delete { node } => {
                        if let Some(name) = self.handles.remove(&node) {
                            calls.push(GeminiCall::DeleteCachedContent {
                                node,
                                cache_name: name,
                            });
                        }
                    }
                }
            }
            calls
        }
    }
}

pub mod anthropic {
    //! Anthropic prompt caching (`cache_control` markers), write-premium regime.

    use crate::engine::{Engine, Observation};
    use crate::trie::CacheState;

    /// A `cache_control` TTL label as the API expects it.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum TtlLabel {
        /// `{"type": "ephemeral"}` — the default 5-minute tier.
        FiveMinutes,
        /// `{"type": "ephemeral", "ttl": "1h"}`.
        OneHour,
    }

    impl TtlLabel {
        /// The wire value for the `ttl` field (`None` means omit it — the
        /// 5-minute tier is the API default).
        pub fn wire_value(self) -> Option<&'static str> {
            match self {
                TtlLabel::FiveMinutes => None,
                TtlLabel::OneHour => Some("1h"),
            }
        }
    }

    /// Where to place one `cache_control` marker in the outgoing request.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Breakpoint {
        /// Place the marker on the content block ending at this token offset
        /// (block-granular: offsets are multiples of the engine's block size).
        pub token_offset: u64,
        /// Which TTL tier to buy at this breakpoint.
        pub ttl: TtlLabel,
    }

    /// The `cache_control` breakpoints to place on this request, ascending by
    /// offset, capped at the provider's breakpoint budget. Derived from the
    /// committed plan ([`Engine::plan`]) intersected with the request's path.
    pub fn breakpoints_for(engine: &Engine, obs: &Observation) -> Vec<Breakpoint> {
        let trie = engine.trie();
        let mut out = Vec::new();
        for &id in &obs.path {
            if let CacheState::Cached { ttl_hours, .. } = trie.node(id).state {
                out.push(Breakpoint {
                    token_offset: trie.tokens(id),
                    ttl: if ttl_hours >= 1.0 {
                        TtlLabel::OneHour
                    } else {
                        TtlLabel::FiveMinutes
                    },
                });
            }
        }
        out.truncate(engine.prices().max_breakpoints);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::anthropic::{breakpoints_for, TtlLabel};
    use super::gemini::{GeminiAdapter, GeminiCall};
    use crate::engine::{Action, Config, Engine};
    use crate::price::PriceSheet;

    #[test]
    fn gemini_adapter_translates_lifecycle_with_handle_registry() {
        let mut ad = GeminiAdapter::new();

        let calls = ad.translate(
            &[Action::Create {
                node: 7,
                tokens: 5120,
                ttl_hours: 2.0 / 4.5,
            }],
            0.0,
        );
        assert_eq!(
            calls,
            vec![GeminiCall::CreateCachedContent {
                node: 7,
                prefix_tokens: 5120,
                ttl_seconds: 1600, // ceil(2/4.5 * 3600)
            }]
        );

        // Extend before registration is dropped (no handle to patch).
        let calls = ad.translate(
            &[Action::Extend {
                node: 7,
                expires_at_hours: 1.0,
            }],
            0.5,
        );
        assert!(calls.is_empty());

        ad.register(7, "cachedContents/abc".into());
        let calls = ad.translate(
            &[Action::Extend {
                node: 7,
                expires_at_hours: 1.0,
            }],
            0.5,
        );
        assert_eq!(
            calls,
            vec![GeminiCall::UpdateExpiry {
                node: 7,
                cache_name: "cachedContents/abc".into(),
                ttl_seconds: 1800,
            }]
        );

        let calls = ad.translate(&[Action::Delete { node: 7 }], 1.0);
        assert_eq!(
            calls,
            vec![GeminiCall::DeleteCachedContent {
                node: 7,
                cache_name: "cachedContents/abc".into(),
            }]
        );
        assert!(ad.handle(7).is_none(), "delete must drop the handle");
    }

    #[test]
    fn anthropic_breakpoints_follow_committed_plan() {
        let mut eng = Engine::new(PriceSheet::anthropic_sonnet_like(), Config::default());
        let blocks: Vec<u64> = (0..10).collect(); // 2560 tokens
        for i in 0..30 {
            eng.observe(&blocks, i as f64 * (2.0 / 60.0));
        }
        let placements = eng.plan(1.0);
        assert!(!placements.is_empty());

        let obs = eng.observe(&blocks, 1.01);
        let bps = breakpoints_for(&eng, &obs);
        assert_eq!(bps.len(), 1);
        assert_eq!(bps[0].token_offset, 2560);
        // 2-minute gaps -> the cheap 5-minute tier, which omits the ttl field.
        assert_eq!(bps[0].ttl, TtlLabel::FiveMinutes);
        assert_eq!(bps[0].ttl.wire_value(), None);
        assert!(bps.len() <= eng.prices().max_breakpoints);
    }
}
