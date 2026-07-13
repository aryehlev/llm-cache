# PCOE: Prefix-Cache Optimization Engine

**An online, cost-model-driven algorithm for managing provider-side explicit prompt caches**

*Design document — v0.1, 2026-07-13*

---

## 1. Problem statement

Modern LLM providers moved the most valuable cache *inside their own infrastructure*.
Instead of caching **responses** (which is either exact-match — low hit rates — or
semantic — probabilistic and unsafe), providers now cache the **computed prefix
state** (the transformer KV cache) of a prompt, and bill for it:

- **Google Gemini — explicit caching.** The developer creates a `CachedContent`
  object holding a prompt prefix (system instructions, documents, media). The cache
  has a TTL, is billed **per token-hour of storage**, and subsequent requests that
  reference it pay a steep discount on those tokens. The developer must decide
  *what* to cache, *when* to create it, *when* to extend its TTL, and *when* to
  delete it.
- **Anthropic — prompt caching.** The developer places up to 4 `cache_control`
  breakpoints in the request. Writes cost a premium (1.25× input price for 5-minute
  TTL, 2× for 1-hour TTL), reads cost ~0.1× input price. The developer must decide
  *where* to place breakpoints and *which TTL tier* to buy.
- **OpenAI — automatic caching.** No explicit control; the only developer lever is
  shaping prompts so shared content forms a stable prefix.

Every one of these is a *pricing-mediated resource-allocation problem*, and today it
is solved by hand: engineers eyeball which prompts are "big and reused," hard-code a
breakpoint, and eat storage costs on prefixes that stopped being reused hours ago.
There is no published algorithm that (a) observes live traffic, (b) models the
provider's actual price sheet, and (c) makes provably-bounded create / extend /
delete / place decisions automatically.

**PCOE is that algorithm.** It sits client-side (library or proxy), watches the
stream of outgoing LLM requests, maintains a privacy-safe statistical model of
prefix reuse, and emits cache-management actions against provider APIs.

### What PCOE is *not*

- Not a response cache (no answers are stored or replayed).
- Not a semantic/vector cache (no embeddings, no similarity thresholds, no
  correctness risk — every decision only changes *cost*, never *output*).
- Not a self-hosted KV-cache store (LMCache/Mooncake territory — see §6); PCOE
  manages *paid provider-side* caches it cannot see inside, only pay for.

---

## 2. Cost model

The algorithm is parameterized by a per-provider price sheet. All decision rules in
§3 are written against these symbols; the tables below instantiate them.

| Symbol | Meaning | Unit |
|---|---|---|
| `p_in` | Base input token price | $/MTok |
| `p_read` | Cached-token read price | $/MTok |
| `p_write(ttl)` | Cache-write price for TTL tier | $/MTok |
| `p_store` | Cache storage price (Gemini-style) | $/MTok/hour |
| `M_min` | Minimum cacheable prefix size | tokens |
| `K_max` | Max breakpoints per request (Anthropic-style) | count |
| `TTL set` | Available TTL tiers | duration |

### 2.1 Anthropic prompt caching (verified 2026-07-13, from Anthropic's current API documentation)

- **Reads: ~0.1× base input price.** **Writes: 1.25× for 5-minute TTL, 2× for
  1-hour TTL.**
- Break-even (write + 1 read vs. uncached): 5-minute TTL pays for itself at **2
  requests** (1.25 + 0.1 = 1.35× vs 2×); 1-hour TTL needs **≥3 requests**
  (2 + 0.2 = 2.2× vs 3×).
- **Max 4 `cache_control` breakpoints per request**; breakpoints may be nested
  (render order `tools → system → messages`, so a deeper breakpoint covers
  everything before it).
- **Minimum cacheable prefix is model-dependent:** 4096 tokens (Opus 4.8/4.7/4.6/4.5,
  Haiku 4.5), 2048 (Fable 5, Sonnet 4.6), 1024 (Sonnet 4.5 and older Sonnets).
  Prefixes below the minimum silently don't cache.
- A cache **read refreshes the entry's TTL at no extra cost** — continuous traffic
  keeps a 5-minute entry alive indefinitely; the TTL only matters across *gaps*.
- Caching is a **prefix byte-match**; any change upstream of a breakpoint
  invalidates it. Caches are model-scoped.
- Reference base prices (per MTok input): Opus 4.8 $5.00, Sonnet 4.6 $3.00,
  Haiku 4.5 $1.00.

### 2.2 Google Gemini explicit caching (secondary sources, accessed 2026-07-13 — Google's first-party docs were unreachable from this environment; **re-verify against [ai.google.dev/gemini-api/docs/caching](https://ai.google.dev/gemini-api/docs/caching) before implementation**)

- **Reads: ~10% of base input price** for cached tokens (e.g. Gemini 3.1 Pro
  $2.00/MTok input → $0.20/MTok cached read).
  ([Morph pricing guide](https://www.morphllm.com/gemini-api-pricing),
  [FindSkill guide](https://findskill.ai/blog/gemini-api-pricing-guide/))
- **Storage: billed per token-hour** — ≈$4.50/MTok/hour for Pro-tier models,
  ≈$1.00/MTok/hour for Flash-tier.
  ([metacto pricing analysis](https://www.metacto.com/blogs/the-true-cost-of-google-gemini-a-guide-to-api-pricing-and-integration))
- **Cache creation** bills the cached tokens once at the standard input rate (no
  premium multiplier, unlike Anthropic — the cost is the storage meter that starts
  running).
- **Default TTL 60 minutes; no hard min/max** — the TTL / `expire_time` can be
  **updated** on a live cache, and a cache can be **deleted** explicitly.
  ([Google Cloud context-cache overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/context-cache/context-cache-overview),
  [Firebase AI Logic docs](https://firebase.google.com/docs/ai-logic/context-caching))
- **Minimum cacheable size is model-dependent**; sources report 1,024–4,096 tokens
  for the mainline API models (with a much higher 32K floor on the enterprise/Vertex
  variant). Treat `M_min` as a per-model config value.
- A request references **one** cache object (the prefix), unlike Anthropic's nested
  breakpoints.

### 2.3 OpenAI (degenerate case)

Automatic prefix caching with no developer-controlled writes, TTLs, or deletes.
Only PCOE's *request-shaping* component (§3.4) applies: maximize the probability
that the automatic cache hits by keeping shared content in a stable prefix.

### 2.4 The two economic regimes

The two pricing designs produce **different optimization problems**, which is why
PCOE has two lifecycle rules that share one traffic model:

| | Anthropic (write-premium regime) | Gemini (storage-meter regime) |
|---|---|---|
| Ongoing cost of a live cache | zero (reads refresh TTL free) | `T · p_store` per hour |
| Cost of losing the cache | re-write premium on next request | re-create at `T · p_in` |
| Core decision | *where* to put ≤4 breakpoints; *which* TTL tier | *when* to create, extend, delete |
| Shape of the problem | tree knapsack (§3.2) | ski-rental (§3.3) |

---

## 3. The algorithm

PCOE has four components sharing one data structure:

```
                 ┌──────────────────────────────────────────┐
 requests ──────▶│ 1. Decayed prefix trie (traffic model)   │
                 └───────┬───────────────────┬──────────────┘
                         │                   │
        ┌────────────────▼───┐      ┌────────▼─────────────┐
        │ 2. Value function +│      │ 3. Lifecycle          │
        │    breakpoint DP   │      │    controller         │
        │  (what & where)    │      │  (create/extend/del)  │
        └────────────────┬───┘      └────────┬─────────────┘
                         │                   │
                 ┌───────▼───────────────────▼──────────────┐
                 │ 4. Request shaping & provider adapters   │
                 └──────────────────────────────────────────┘
```

### 3.1 Traffic model: decayed prefix trie over chunk hashes

**Representation.** Each outgoing request's prompt is split into fixed-size token
blocks (default `B = 256` tokens; the last partial block is dropped from the key —
it can never be a stable prefix boundary). Each block is hashed (any strong 128-bit
hash). The sequence of block hashes is inserted as a path into a trie. **No raw
prompt content is retained** — the trie stores only hashes, token counts, and
timestamps, making the telemetry safe to persist and ship.

**Per-node statistics.** Each trie node `n` maintains:

| Field | Meaning |
|---|---|
| `T(n)` | Cumulative token count from root to `n` (depth × B) |
| `λ̂(n)` | Exponentially-decayed arrival rate estimate (requests/hour through this prefix) |
| `t_last(n)` | Timestamp of last traversal |
| `gaps(n)` | Decayed histogram of inter-arrival gaps (log-spaced buckets: <5m, 5m–1h, >1h) |
| `state(n)` | `UNCACHED` \| `CACHED(provider_handle, expires_at, tier)` \| `PENDING` |

**Rate update (lazy decay).** On each traversal of node `n` at time `t`:

```
λ̂(n) ← λ̂(n) · exp(−(t − t_last(n)) / τ_decay) + 1/τ_decay
gaps(n).add(t − t_last(n))          # same decay applied to histogram mass
t_last(n) ← t
```

with `τ_decay` a half-life-style smoothing constant (default 1 hour). This is a
standard EWMA rate estimator; decay is applied lazily on access so idle nodes cost
nothing. Nodes with `λ̂ < λ_gc` and `t_last` older than `τ_gc` are garbage-collected;
the trie is additionally capped at `N_max` nodes with least-λ̂ eviction, so memory is
bounded regardless of traffic diversity.

**Why a trie and not per-prompt counters:** reuse is *hierarchical*. A support bot's
traffic shares (a) the system prompt across all users, (b) system + policy docs
across one product line, (c) system + docs + conversation history within one
session. These are nested prefixes with different reuse rates and different optimal
cache decisions — exactly a root-to-leaf path in the trie with λ̂ decreasing (weakly)
with depth. Note the invariant: **λ̂(parent) ≥ λ̂(child)** (every traversal of a
child traverses its parent), which both the DP and the lifecycle rules exploit.

### 3.2 Value function and breakpoint placement (the optimization core)

**Value of caching a node.** If node `n` is cached and a request traverses it (and
`n` is the deepest cached node on that request's path — see below), the request pays
`p_read` instead of `p_in` on `T(n)` tokens. Expected gross savings rate:

```
S(n) = λ̂(n) · T(n)/1e6 · (p_in − p_read)          [$/hour]
```

Cost rate of keeping `n` cached (regime-dependent):

```
C(n) = T(n)/1e6 · p_store                               (storage regime)
C(n) = (rewrites/hour)(n) · T(n)/1e6 · (p_write − p_in) (write-premium regime)
```

where `(rewrites/hour)(n)` is estimated from `gaps(n)`: the decayed rate of
inter-arrival gaps longer than the chosen TTL (each such gap forces one cold
rewrite).

**Net value:** `V(n) = S(n) − C(n)`. A node is *cache-worthy* iff `V(n) > 0` and
`T(n) ≥ M_min`.

**The placement problem (Anthropic-style, ≤ K_max breakpoints).** Caching decisions
interact: if both `n` and its descendant `d` are cached, a request through `d` reads
via `d` (covering `T(d)` tokens), so `n` only earns savings on traffic that passes
through `n` **but not** through any cached descendant. Choosing the best set of
≤ K_max nodes is a **tree knapsack**. Define for each node the *exclusive rate*
given a chosen set `A`:

```
λ_excl(n | A) = λ̂(n) − Σ λ̂(d)   over d ∈ A, d a highest cached descendant of n
```

Maximize `Σ_{n∈A} [ λ_excl(n|A) · T(n)/1e6 · (p_in − p_read) − C(n) ]` subject to
`|A| ≤ K_max`. Because savings compose along root-to-leaf paths, this decomposes
into a standard DP over the trie:

```
# f(n, k, covered) = best net value in subtree(n) using k breakpoints,
# where `covered` ∈ {true,false} says whether some ancestor of n is cached.
# (`covered` matters only through C(n)'s rewrite estimation on the write-premium
#  regime; on the storage regime it can be dropped.)

f(leaf, k, c)  = max(0, place_here(leaf, c)) if k ≥ 1 else 0
f(n, k, c) = max over (k_self ∈ {0,1}, partition of k − k_self among children):
    k_self · value(n, c)                    # value uses λ_excl via child subtraction
    + Σ_child f(child, k_child, c ∨ k_self)
```

With `N` trie nodes and `K_max ≤ 4` this is `O(N · K_max²)` per re-solve — trivial
at the scale of real traffic tries (thousands of nodes). Re-solve on a timer
(default every 60 s) and on any λ̂ change that crosses a hysteresis band (below).

**Hysteresis (anti-thrash).** A node's state flips only if the decision persists:
promote to cached when `V(n) > θ_up · C(n)` and demote when `V(n) < θ_down · C(n)`
with `θ_up = 1.2, θ_down = 0.8` by default, and a minimum dwell time of one TTL
period between flips of the same node. Without this, λ̂ noise around the break-even
point causes write-premium churn that itself costs money.

**TTL-tier selection (Anthropic).** For each candidate node, estimate rewrite rates
under each tier from `gaps(n)`:

```
choose 1h tier  iff  gaps-rate(5m < gap ≤ 1h) · (p_write_1h − p_write_5m are both premiums):
    cost_5m = rate(gap > 5m)  · T · (0.25 · p_in)     # 1.25× − 1×
    cost_1h = rate(gap > 1h)  · T · (1.00 · p_in)     # 2× − 1×
    pick min(cost_5m, cost_1h)
```

i.e. buy the expensive 1-hour tier only when the traffic's gap histogram shows
enough 5-minute-to-1-hour gaps that repeated 5-minute rewrites cost more than the
one-time 2× write. This is where the gap *histogram* (not just the mean rate)
earns its keep: two workloads with identical λ̂ but different burstiness get
different tiers.

### 3.3 Lifecycle controller: ski-rental with learned rates (storage regime)

For Gemini-style caches the running decision is *hold vs. delete* (and its mirror,
*create vs. wait*), which is the classical **ski-rental / rent-to-buy** problem:
storage is rent, (re)creation at `T · p_in` is the buy price.

**Deletion rule (worst-case-safe).** After the last hit on cached node `n`, keep
paying storage until cumulative storage spend equals the recreation cost, then
delete:

```
τ_hold = (T · p_in) / (T · p_store) = p_in / p_store     [hours]
```

Note `T` cancels — **the optimal hold time is independent of the cache's size** and
depends only on the price ratio. At reported Gemini 3.1 Pro prices
(`p_in = $2.00/MTok`, `p_store = $4.50/MTok/h`): `τ_hold ≈ 26.7 minutes`. This is
the deterministic break-even ski-rental strategy, with the classical worst-case
guarantee: **total cost ≤ 2× the offline optimum** regardless of the (adversarial)
arrival sequence [Karlin et al., *Competitive snoopy caching*, Algorithmica 1988].
The randomized variant (delete at a random time drawn from the exponential-tilted
distribution over `[0, τ_hold]`) improves the bound to `e/(e−1) ≈ 1.58`.

**Rate-informed override.** When `λ̂(n)` is confident (decayed sample count above a
floor), switch from worst-case to expected-value: hold iff the expected next-hit
time `1/λ̂(n)` satisfies

```
hold  iff  E[storage until next hit] < P(hit before abandon) · recreation savings
       ⟺  p_store / λ̂(n) < p_in − p_read·(...)      (simplified: 1/λ̂ < τ_hold)
```

i.e. hold while the expected gap is shorter than `τ_hold`, delete immediately when
the estimated regime says the next hit will arrive after break-even anyway. The
worst-case rule remains a fallback whenever the estimator is cold — so the
2-competitive guarantee is never lost, only improved upon. (This
predictor-plus-worst-case-fallback structure follows the *learning-augmented
ski-rental* line of work [Purohit, Svitkina & Kumar, NeurIPS 2018].)

**Creation rule.** On a request whose deepest cache-worthy node `n` is `UNCACHED`:

```
create  iff  λ̂(n) · τ_hold ≥ 1 + margin        # expect ≥1 more hit inside the hold window
```

(the request itself proceeds uncached either way — creation is fire-and-forget and
never blocks the request path). While the create call is in flight the node is
`PENDING` so concurrent requests don't duplicate it.

**Extension rule (TTL update).** Gemini permits updating `expire_time` on a live
cache. Rather than buying long TTLs up front, PCOE creates with a short TTL and
extends just before expiry iff the hold rule still says *hold*. Extension is free
apart from the storage it commits to, so the extend decision *is* the hold decision
evaluated at `expires_at − ε`.

### 3.4 Request shaping & scheduling

These components *increase the hit rate the other components can harvest*. They are
also the only levers on automatic-caching providers (OpenAI).

**a) Prefix-stability canonicalization.** Volatile fields destroy prefix reuse when
they render early. PCOE classifies each prompt segment by observed volatility
(a segment whose block hashes differ across otherwise-identical trie paths is
volatile) and — where the integration layer permits reordering — moves volatile
segments (timestamps, request IDs, user-specific footers) to the suffix, after the
last stable block. This is the automated version of the "keep the system prompt
frozen / put volatile content last" guidance that today lives in provider docs as
manual advice. The trie itself is the detector: a node with high fan-out
(many single-traversal children) marks the stable→volatile boundary; PCOE emits
the boundary as a recommended breakpoint position *and* a lint ("segment X at
offset Y varies per request; move after Z to unlock $W/hour of savings").

**b) Write-amortizing micro-batching.** When the deepest cache-worthy node for an
arriving request is `UNCACHED` and the trie predicts `k = λ̂(n) · Δ` further
arrivals within a short window `Δ` (default ≤ 2 s, opt-in), PCOE dispatches the
first request immediately, initiates the cache write, and briefly defers the
*followers* until the write is live — so `k` requests pay one write premium
instead of `k`. On providers where a cache entry becomes readable only after the
first response begins streaming, the follower-deferral is until first-token of the
leader. Deferral never exceeds `Δ` (bounded added latency, off by default).

**c) Cross-provider routing (optional).** When an application is provider-agnostic
for a request family, the same value function evaluated per provider yields the
cheaper venue *including cache state* — e.g. a family with an already-live Anthropic
cache routes there even if Gemini's marginal price looks lower uncached. Routing
decisions feed back into the trie (per-provider `state`), making this a small
extension rather than a separate system.

### 3.5 Safety properties

- **Output-invariance:** no PCOE action can change a model's output; caches are
  bit-exact prefix state on the provider side. Worst case of a wrong decision is
  paying more, bounded by the ski-rental guarantee (storage regime) or one write
  premium per hysteresis flip (write-premium regime).
- **Fail-open:** if the controller or provider cache API errors, requests proceed
  uncached.
- **Privacy:** the trie stores block hashes and counters only; no prompt text.

---

## 4. Worked numeric example

*Scenario:* a document-QA assistant. Shared prefix = system prompt + product corpus
= **200K tokens**. Traffic: **12 requests/hour for 8 business hours**, zero
overnight (96 requests/day). Suffixes (user questions) are unique per request and
never cacheable.

### 4.1 Storage regime (Gemini-3.1-Pro-style prices: `p_in=$2.00`, `p_read=$0.20`/MTok, `p_store=$4.50`/MTok/h)

Per-day cost of the 200K-token prefix under four policies
(computed and machine-checked; see §7 Verification):

| Policy | Create | Reads | Storage | **Total/day** | vs. no cache |
|---|---:|---:|---:|---:|---:|
| **No caching** | — | 96 × 0.2 MTok × $2.00 = $38.40 | — | **$38.40** | — |
| **Cache-everything, hold 24 h** | $0.40 | 96 × 0.2 × $0.20 = $3.84 | 24 h × 0.2 × $4.50 = $21.60 | **$25.84** | −33% |
| **Static hand-tuning** (hold business hours + 1 h pad) | $0.40 | $3.84 | 9 h × $0.90 = $8.10 | **$12.34** | −68% |
| **PCOE** (rate-informed hold; delete τ_hold=26.7 min after last hit) | $0.40 | $3.84 | ≈ 8.4 h × $0.90 = $7.60 | **$11.84** | **−69%** |

Notes: during business hours the expected gap is `1/λ̂ = 5 min < τ_hold = 26.7 min`,
so PCOE holds continuously (storage $0.90/h against gross savings
`12 × 0.2 × 1.8 = $4.32/h` — clearly positive). At close of business the gap
estimator (or, cold, the ski-rental rule) triggers deletion within ~27 minutes;
overnight the cache is gone, saving the $14+ that cache-everything burns. Next
morning's first request re-creates for $0.40. PCOE ≈ matches the *best possible*
hand tuning without anyone having to know the traffic schedule — and when the
schedule drifts (a new region's business hours, a weekend), the hand-tuned config
silently degrades while PCOE re-learns within `τ_decay`.

The same arithmetic at Flash-tier storage (`$1.00/MTok/h`) stretches
`τ_hold` to `p_in/p_store` hours — cheap storage means hold much longer; the rule
adapts with zero code change.

### 4.2 Write-premium regime (Anthropic-style: Sonnet 4.6, `p_in=$3.00`, `p_read=$0.30`, write 1.25×/2.0×)

Same traffic. The gap histogram shows: intra-day gaps ≈ 5 min (just at the 5-minute
TTL edge — reads refresh the TTL free, so a 5-minute tier survives the day iff gaps
stay < 5 min; with Poisson arrivals at λ=12/h, P(gap > 5 min) = e^(−1) ≈ 37%, i.e.
≈ 35 cold rewrites/day on the 5-minute tier) versus one overnight gap.

| Policy | Writes/day | Write cost | Reads | **Total/day** |
|---|---:|---:|---:|---:|
| No caching | — | — | 96 × 0.2 × $3.00 = $57.60 | **$57.60** |
| 5-min tier | ≈ 36 | 36 × 0.2 × $3.75 = $27.00 | 60 × 0.2 × $0.30 = $3.60 | **$30.60** |
| **1-h tier (PCOE picks this)** | ≈ 2 | 2 × 0.2 × $6.00 = $2.40 | 94 × 0.2 × $0.30 = $5.64 | **$8.04 (−86%)** |

The tier decision flips on the gap histogram, not the mean rate — exactly the
statistic §3.2 keeps. (A workload with the same λ̂ but tight 2-minute gaps would
flip the table: the 5-minute tier would see ~1 rewrite/day and win.) PCOE also
emits the breakpoint placement: one breakpoint after the corpus (the trie node
where fan-out explodes), leaving the unique question in the uncached suffix —
using 1 of the 4 available breakpoints and reserving the rest for deeper
conversation-history nodes if multi-turn traffic appears.

---

## 5. Provider adapters

The core emits abstract actions; adapters translate:

| Abstract action | Gemini adapter | Anthropic adapter | OpenAI adapter |
|---|---|---|---|
| `CREATE(node, ttl)` | `cachedContents.create` with prefix content, `ttl` | (implicit — next request carries `cache_control` at the node boundary with chosen tier) | — |
| `EXTEND(node, t)` | `cachedContents.patch` `expire_time` | (no-op — reads refresh free) | — |
| `DELETE(node)` | `cachedContents.delete` | (no-op — stop marking; entry ages out) | — |
| `PLACE(breakpoints)` | choose which single ancestor cache the request references | insert ≤4 `cache_control` markers at chosen block boundaries | — |
| `SHAPE(reorder)` | ✓ | ✓ | ✓ (only lever) |

Adapter-specific constraints enforced at this layer: Gemini's one-cache-per-request
and per-model `M_min`; Anthropic's `K_max = 4`, per-model `M_min`, 20-block
lookback window (PCOE inserts an intermediate breakpoint every ~15 blocks in long
agentic turns), and model-scoped caches (a model switch resets the node's `state`).

---

## 6. Prior art and novelty claims

### 6.1 Adjacent work (and why it is *not* this)

| Work | Layer | Relationship |
|---|---|---|
| **SGLang RadixAttention** — [Zheng et al., arXiv:2312.07104](https://arxiv.org/abs/2312.07104) | Self-hosted GPU serving | Maintains a radix tree over token sequences to reuse KV cache **inside the server**, evicting by LRU. Establishes prior art for *radix/prefix trees over prompt traffic*. It optimizes GPU memory it owns and can see; it has no monetary cost model, no TTL purchases, no provider API to manage. |
| **Preble** — [Srivatsa et al., arXiv:2407.00023](https://arxiv.org/abs/2407.00023) (ICLR 2025) | Distributed self-hosted serving | Prefix-aware *scheduling*: routes requests across GPUs to co-locate shared prefixes. Prior art for *prefix-aware request routing*. Again self-hosted, throughput-optimizing, no pricing. |
| **LMCache / Mooncake** | Self-hosted KV storage tiers | Store KV tensors in RAM/SSD/remote for vLLM-class servers. Different layer entirely (they hold the actual tensors). |
| **Prompt Cache** — [Gim et al., arXiv:2311.04934](https://arxiv.org/abs/2311.04934) | Model-level | Reuses attention states of *modular prompt segments*. Requires model access. |
| **Ski-rental / snoopy caching** — Karlin, Manasse, Rudolph & Sleator, *Algorithmica* 1988; learning-augmented variant Purohit, Svitkina & Kumar, NeurIPS 2018 | Online-algorithms theory | The rent-vs-buy machinery §3.3 instantiates. Generic theory; no application to priced LLM prompt caches. |
| **TTL-cache economics** (CDN literature; e.g. utility-driven TTL caching, Dehghan et al. 2019) | Networking | Optimizes TTLs for object caches under hit-rate utility. Not prefix-structured, not two-regime priced. |
| **GPTCache and semantic caches** | Client-side response caching | Cache *answers* by embedding similarity — probabilistic correctness. PCOE caches nothing content-bearing and cannot change outputs. |
| **Provider docs** (Anthropic prompt-caching guide, Gemini caching guide) | Documentation | Give *manual* heuristics ("put stable content first", break-even tables). No algorithm, no automation, no traffic model. |

### 6.2 Claimed novel combination

To our knowledge, no prior system or publication combines:

1. A **client-side, privacy-safe decayed prefix trie** built from *hashed block
   sequences of live API traffic* (not server-side tensors), annotated with
   decayed arrival rates **and inter-arrival gap histograms** per node;
2. A **provider price-sheet-parameterized value function** over that trie,
   covering both pricing regimes (per-token-hour storage vs. write-premium + TTL
   tiers), with the closed-form size-independent hold rule
   `τ_hold = p_in / p_store`;
3. A **budgeted tree-knapsack DP** selecting ≤ K nested cache breakpoints under
   exclusive-coverage semantics, with hysteresis bands sized to write premiums;
4. A **learning-augmented ski-rental lifecycle controller** issuing create /
   extend / delete calls against a *paid third-party* cache API, retaining the
   2-competitive worst-case bound when the rate estimator is cold;
5. **Trie-driven prompt canonicalization and write-amortizing micro-batching**
   that reshape and reschedule requests to raise the hit rate the controller can
   harvest, including cross-provider routing on cache-state-aware marginal cost.

The individually-known ingredients (radix trees over prompts — RadixAttention;
ski-rental — Karlin et al.; prefix-aware scheduling — Preble) operate at other
layers with other objectives; the claim is the *system*: economic lifecycle control
of provider-side explicit prompt caches driven by a client-observable traffic
model. Elements (2)'s two-regime formulation, (3)'s breakpoint DP with gap-histogram
TTL-tier selection, and (5)'s deferral-for-write-amortization appear to have no
direct precedent at any layer.

**Caveat:** this table is an engineering survey, not a legal freedom-to-operate or
patentability search. See §8.

---

## 7. Verification

- Anthropic pricing figures (§2.1) are taken from Anthropic's current prompt-caching
  documentation as bundled in Anthropic's own developer tooling on 2026-07-13
  (write premiums 1.25×/2×, ~0.1× reads, 4-breakpoint limit, per-model minimums,
  free TTL refresh on read).
- Gemini figures (§2.2) come from the cited secondary sources on 2026-07-13 because
  Google's first-party docs were unreachable from this environment; they match
  across three independent sources but **must be re-verified against
  ai.google.dev before implementation or filing**.
- The §4 tables were computed by script (not hand arithmetic):
  totals $38.40 / $25.84 / $12.34 / $11.84 per day (storage regime) and
  $57.60 / $30.60 / $8.04 (write-premium regime, with the Poisson
  P(gap > 5 min) = e^(−1) rewrite estimate) — reproducible from the formulas in
  §3–§4 with the stated prices.
- Prior-art citations were checked against the arXiv abstracts/landing pages on
  2026-07-13: RadixAttention is arXiv:2312.07104; Preble is arXiv:2407.00023
  (ICLR 2025); Prompt Cache is arXiv:2311.04934.

## 8. Future work

- **Self-hosted tier:** the same trie + value function can drive eviction in a
  local KV-cache store (RAM → SSD), replacing `p_store` with hardware amortization —
  bridging PCOE to the LMCache/Mooncake layer with one cost model.
- **Reference implementation:** Rust library (`llm-cache` crate): trie + controller
  as a pure core, provider adapters behind a trait, proxy binary later.
- **Simulation harness:** replay real (hashed) traffic traces through the
  controller vs. oracle-offline-optimal to measure the empirical competitive ratio.
- **Richer predictors:** periodicity detection (daily/weekly seasonality) on top of
  the EWMA — the office-hours pattern in §4 is learnable, letting PCOE pre-create
  the cache at 8:59 rather than paying one cold miss.

## 9. Disclaimer

This is an engineering design document, not legal advice. The prior-art survey in
§6 is informal. Before any patent filing, engage a patent attorney and commission a
professional prior-art / freedom-to-operate search; claims should be drafted by
counsel from §6.2 as raw material.
