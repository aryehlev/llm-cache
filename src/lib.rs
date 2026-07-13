//! # PCOE — Prefix-Cache Optimization Engine
//!
//! An online, cost-model-driven controller for *provider-side explicit prompt
//! caches* (Gemini `CachedContent`, Anthropic `cache_control`). PCOE watches the
//! stream of outgoing LLM requests, maintains a privacy-safe statistical model of
//! prefix reuse (a decayed prefix trie over hashed token blocks — no prompt text
//! is retained), and emits cache-management actions: what to cache, where to place
//! breakpoints, which TTL tier to buy, and when to create, extend, or delete.
//!
//! See `DESIGN.md` in the repository root for the full algorithm specification.
//!
//! ## Time model
//!
//! All APIs take `now` as `f64` **hours** on any monotonic clock of the caller's
//! choosing (e.g. hours since process start). This keeps the core deterministic
//! and testable; nothing in the crate reads the system clock.
//!
//! ## Example (storage-metered regime, Gemini-like prices)
//!
//! ```
//! use llm_cache::{Config, Engine, PriceSheet};
//!
//! let mut engine = Engine::new(PriceSheet::gemini_pro_like(), Config::default());
//! // 20 blocks x 256 tokens = 5120 tokens of stable shared prefix.
//! let blocks: Vec<u64> = (0..20).collect();
//!
//! let mut created = false;
//! for i in 0..10 {
//!     let now = i as f64 * (5.0 / 60.0); // a request every 5 minutes
//!     let obs = engine.observe(&blocks, now);
//!     created |= !obs.actions.is_empty();
//! }
//! // Once the arrival-rate estimate clears break-even, PCOE asks for a cache.
//! assert!(created);
//! ```

pub mod chunk;
pub mod engine;
pub mod plan;
pub mod price;
pub mod trie;

pub use engine::{Action, Config, Engine, Observation};
pub use plan::Placement;
pub use price::{PriceSheet, Regime, TtlTier};
pub use trie::{CacheState, NodeId, PrefixTrie};
