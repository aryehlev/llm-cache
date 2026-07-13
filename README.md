# llm-cache

Design work for **PCOE (Prefix-Cache Optimization Engine)** — an online,
cost-model-driven algorithm that automatically manages *provider-side explicit
prompt caches* (Gemini `CachedContent`, Anthropic `cache_control`): what to cache,
where to place breakpoints, which TTL tier to buy, and when to create, extend, or
delete — driven by a privacy-safe decayed prefix trie of live traffic and the
provider's actual price sheet.

See **[DESIGN.md](./DESIGN.md)** for the full algorithm specification, cost model,
worked cost examples, prior-art survey, and novelty claims.
