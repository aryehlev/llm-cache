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
| `engine` | §3.3 | Ski-rental lifecycle controller emitting `Create` / `PreCreate` / `Extend` / `Delete` actions with pending-create confirmation (`confirm_create` / `mark_failed` / timeout), a **learned per-node hold** that self-corrects premature deletes, **predictive pre-creation** that warms a cache just before a confidently-periodic return (daily opens, per-tenant windows) — both zero-regression on unpredictable traffic — dominated-ancestor retirement, fail-open. |
| `shape` | §3.4a | Stable→volatile boundary detection from trie fan-out: where the cacheable prefix ends, and what a cache point there would earn. |
| `engine` (micro-batch) | §3.4b | Opt-in write-amortizing micro-batching: followers of an in-flight cache write get bounded `Defer` advice so one write premium covers the batch. |
| `router` | §3.4c | Cross-provider routing on cache-state-aware marginal input cost (read-only quotes that don't pollute traffic stats). |
| `adapter` | §5 | Wire-agnostic Gemini (`cachedContents` create/patch/delete + handle registry) and Anthropic (`cache_control` breakpoint offsets + TTL labels) translation layers. |
| `chunk` | §3.1 | `Chunker` trait for pluggable tokenizers, plus token-ID and byte-block hashing helpers. |
| `trace` / `sim` | §8 | Traffic traces (privacy-safe: block hashes only) + a replay simulator that costs PCOE against every baseline and an offline oracle. |

The engine is deterministic and clock-free (callers pass `now` in hours), talks to
no provider directly (adapters translate actions into concrete API operations the
application executes with its own content — PCOE never holds prompt text), and is
fail-open by construction — a wrong decision can only cost money, never change
model output.

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

## Proving ROI: `pcoe-sim`

The point of a cost optimizer is the dollar figure on *your* traffic, not a
synthetic claim. `pcoe-sim` replays a traffic trace — a captured production one
(privacy-safe: block hashes only, no prompt text) or a synthetic workload —
through PCOE and every alternative, and prints what each would have spent,
including an **offline oracle** (perfect-hindsight ski-rental) to bound how much
headroom remains.

```sh
cargo run --bin pcoe-sim -- run --scenario business-day --days 7
```

```text
  PCOE ROI report — scenario business-day (7 days, seed 42)
  provider: gemini (storage-metered)   requests: 670

  policy                         cost ($)    vs no-$  hit rate ops(c/e/d)
  ----------------------------------------------------------------------
  no caching                       268.60          —        0%          —
  cache-everything                 164.73       -39%      100%      1/0/0
  static TTL (60m idle)             87.34       -67%       99%      7/0/6
▶ PCOE                              87.28       -68%       96%   11/156/8
  offline oracle                    79.32       -70%         —          —

  PCOE vs no caching:       -68%   ($268.60 -> $87.28)
  PCOE vs cache-everything: -47%   ($164.73 -> $87.28)
  PCOE vs offline optimal:  1.10x  (1.00x = perfect hindsight)
```

Capture a trace from production and replay it:

```sh
pcoe-sim gen multi-tenant --days 7 --out fleet.trace   # or write your own
pcoe-sim run --trace fleet.trace
```

### Where it helps — and where it doesn't (measured, not claimed)

The simulator is honest about PCOE's envelope. Over a 7-day run at Gemini-Pro-like
prices:

| Workload | PCOE vs no-cache | PCOE vs cache-everything | vs offline optimal |
|---|---|---|---|
| **business-day** (one big prefix, nightly idle) | **−68%** | −47% | 1.10× |
| **multi-tenant** (12 staggered tenants) | **−71%** | −47% | 1.11× |
| **bursty** (bimodal hot/cold) | **−77%** | +3% | ≈1.25× (mean) |

- **Strong win** where a big prefix goes idle (nights, weekends) or where many
  tenants have their own hours — cases no single hand-set TTL covers. This is the
  product's core: Gemini storage-metered caching across many tenants.
- **Competitive** with a *well-tuned* static TTL on clean single-prefix traffic —
  PCOE lands within ~6% of a hindsight-perfect hand-tuned TTL. Its value is
  adapting when no single TTL fits, not beating a hand-tuned one on a
  metronome-regular workload.
- **Predictive pre-creation** is the one lever that beats the online ski-rental
  bound, by exploiting *learned periodicity* the adversarial model doesn't have.
  When a prefix's post-delete returns are confidently periodic (a daily open, a
  per-tenant window), PCOE pre-creates the cache a few minutes before the next
  return so the first request reads warm instead of paying the cold-start miss —
  the dominant PCOE-vs-oracle gap on per-tenant fleets. On multi-tenant it cuts
  the ratio **1.135× → 1.111×** (~$12/week/12-tenants at these prices). It is
  strictly **zero-regression**: the confidence gate (≥3 clean cycles, gap
  coefficient-of-variation ≤ 0.05, only idle-and-return cycles past
  `tau_effective`) sits an order of magnitude below where merely-bursty traffic
  registers, so on the bursty workload it never fires (Δ = $0.00 across all 8
  seeds). Pinned by tests.
- **Bimodal bursty traffic is the hardest case.** A purely greedy controller
  deletes during a cold lull right before the next burst and *loses* to
  cache-everything (competitive ratio ≈1.29). The **learned per-node hold** (a
  prefix that keeps getting hit right after a delete grows its hold and rides
  through the lulls) recovers most of that — mean ratio ≈1.25. Because bursty
  traffic isn't periodic, pre-creation stays off here by design; the residual gap
  is the part that genuinely needs *burst prediction* (DESIGN.md §8). Both the
  improvement and the residual headroom are pinned by tests.

**Decision rule:** worth deploying if you spend enough on LLM input tokens that a
large shared prefix sits idle for meaningful stretches, or you manage many tenants
— especially on Gemini. The loss is bounded (worst case 2× the offline optimum,
never a wrong answer), so it's safe to A/B against your current setup.
