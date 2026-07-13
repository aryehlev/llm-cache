# llm-cache

Design work for **PCOE (Prefix-Cache Optimization Engine)** — an online,
cost-model-driven algorithm that automatically manages *provider-side explicit
prompt caches* (Gemini `CachedContent`, Anthropic `cache_control`): what to cache,
where to place breakpoints, which TTL tier to buy, and when to create, extend, or
delete — driven by a privacy-safe decayed prefix trie of live traffic and the
provider's actual price sheet.

See **[DESIGN.md](./DESIGN.md)** for the full algorithm specification, cost model,
worked cost examples, prior-art survey, and novelty claims.

## Implementation

A zero-dependency Rust library implementing the PCOE core:

| Module | DESIGN.md | What it does |
|---|---|---|
| `trie` | §3.1 | Decayed prefix trie over hashed token blocks: EWMA arrival rates, inter-arrival gap histograms, GC, node budget. No prompt content retained. |
| `price` | §2 | Two-regime price sheets (storage-metered / write-premium) with `gemini_pro_like()` and `anthropic_sonnet_like()` presets; `tau_hold = p_in / p_store`. |
| `plan` | §3.2 | Value function and budgeted tree-knapsack DP for ≤ K breakpoint placement under exclusive coverage, with gap-histogram TTL-tier selection and hysteresis. |
| `engine` | §3.3 | Ski-rental lifecycle controller emitting `Create` / `Extend` / `Delete` actions, rate-informed early exit, dominated-ancestor retirement, fail-open. |
| `chunk` | §3.1 | Token/byte block hashing helpers. |

The engine is deterministic and clock-free (callers pass `now` in hours), talks to
no provider directly (adapters execute the emitted actions), and is fail-open by
construction — a wrong decision can only cost money, never change model output.

```rust
use llm_cache::{Config, Engine, PriceSheet};

let mut engine = Engine::new(PriceSheet::gemini_pro_like(), Config::default());
// For each outgoing request: chunk the prompt, observe, execute actions.
let obs = engine.observe(&blocks, now_hours);
// Periodically (at least every Config::extend_lead_hours):
let actions = engine.tick(now_hours);
```

Run the tests — `tests/worked_example.rs` replays the DESIGN.md §4.1 business-day
scenario end-to-end and checks the day cost lands near the doc's $11.84 figure:

```sh
cargo test
```
