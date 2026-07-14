//! Provider price sheets — the cost model every decision rule is written against.
//!
//! Two pricing regimes exist in the wild (DESIGN.md §2.4):
//!
//! - **Storage-metered** (Gemini explicit caching): a live cache is billed per
//!   token-hour; creation bills the tokens once at the input rate. The core
//!   decision is *when to create / extend / delete* — a ski-rental problem.
//! - **Write-premium** (Anthropic prompt caching): writes cost a TTL-tiered
//!   premium, holding is free (reads refresh the TTL). The core decision is
//!   *where to place ≤ K breakpoints and which tier to buy*.

/// One TTL tier of a write-premium provider (e.g. Anthropic 5-minute / 1-hour).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TtlTier {
    /// Cache lifetime bought by a write at this tier, in hours.
    pub ttl_hours: f64,
    /// Write price as a multiple of the base input price (e.g. 1.25, 2.0).
    pub write_multiplier: f64,
}

/// Which optimization problem a price sheet induces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    /// Per-token-hour storage billing (Gemini-style): ski-rental lifecycle.
    StorageMetered,
    /// TTL-tiered write premiums (Anthropic-style): breakpoint placement + tier choice.
    WritePremium,
    /// No developer-controlled cache (OpenAI-style): only request shaping applies.
    ShapeOnly,
}

/// A provider's cache-relevant prices and constraints.
#[derive(Clone, Debug)]
pub struct PriceSheet {
    /// Base input token price, $/MTok.
    pub input_per_mtok: f64,
    /// Cached-token read price, $/MTok.
    pub cached_read_per_mtok: f64,
    /// Storage price, $/MTok/hour. `Some` selects the storage-metered regime.
    pub storage_per_mtok_hour: Option<f64>,
    /// Available write tiers. Non-empty (with no storage price) selects the
    /// write-premium regime.
    pub write_tiers: Vec<TtlTier>,
    /// Minimum cacheable prefix size in tokens (`M_min`); smaller prefixes
    /// silently don't cache on real providers.
    pub min_cacheable_tokens: u64,
    /// Maximum simultaneous cache points per request (`K_max`); Anthropic: 4,
    /// Gemini: 1 (one `CachedContent` reference per request).
    pub max_breakpoints: usize,
}

impl PriceSheet {
    /// Which regime this sheet induces.
    pub fn regime(&self) -> Regime {
        if self.storage_per_mtok_hour.is_some() {
            Regime::StorageMetered
        } else if !self.write_tiers.is_empty() {
            Regime::WritePremium
        } else {
            Regime::ShapeOnly
        }
    }

    /// The ski-rental hold time after the last hit, in hours (storage regime).
    ///
    /// `tau = p_in / p_store` — independent of cache size (DESIGN.md §3.3), with
    /// the classical 2-competitive worst-case guarantee. This is the default
    /// and the floor: a cache is never deleted before this.
    pub fn tau_hold_hours(&self) -> Option<f64> {
        self.storage_per_mtok_hour.map(|s| self.input_per_mtok / s)
    }

    /// The upper bound on adaptive holding, in hours (storage regime).
    ///
    /// Losing a warm cache costs recreation (`p_in`) **and** a miss on the
    /// request that finds it cold (`p_in − p_read` extra), so the true
    /// ski-rental *buy* cost is `2·p_in − p_read` and holding can be worth it
    /// up to `tau = (2·p_in − p_read) / p_store`. The controller only holds
    /// this long for a node whose traffic has *demonstrated* it recurs after a
    /// lull (see the learned per-node hold in the engine); the common case
    /// stays at [`Self::tau_hold_hours`]. Still cache-size-independent.
    pub fn tau_effective_hours(&self) -> Option<f64> {
        self.storage_per_mtok_hour
            .map(|s| (2.0 * self.input_per_mtok - self.cached_read_per_mtok) / s)
    }

    /// Gemini-Pro-like storage-metered sheet ($2.00 in / $0.20 cached read per
    /// MTok, $4.50/MTok/h storage). Figures from secondary sources as of
    /// 2026-07-13 — re-verify before production use (DESIGN.md §2.2).
    pub fn gemini_pro_like() -> Self {
        PriceSheet {
            input_per_mtok: 2.00,
            cached_read_per_mtok: 0.20,
            storage_per_mtok_hour: Some(4.50),
            write_tiers: Vec::new(),
            min_cacheable_tokens: 4096,
            max_breakpoints: 1,
        }
    }

    /// Anthropic-Sonnet-4.6-like write-premium sheet ($3.00 in / $0.30 cached
    /// read per MTok; 1.25x write for 5-minute TTL, 2x for 1-hour; 4 breakpoints;
    /// 2048-token minimum). Verified against Anthropic docs 2026-07-13.
    pub fn anthropic_sonnet_like() -> Self {
        PriceSheet {
            input_per_mtok: 3.00,
            cached_read_per_mtok: 0.30,
            storage_per_mtok_hour: None,
            write_tiers: vec![
                TtlTier {
                    ttl_hours: 5.0 / 60.0,
                    write_multiplier: 1.25,
                },
                TtlTier {
                    ttl_hours: 1.0,
                    write_multiplier: 2.0,
                },
            ],
            min_cacheable_tokens: 2048,
            max_breakpoints: 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tau_hold_is_price_ratio() {
        let p = PriceSheet::gemini_pro_like();
        let tau = p.tau_hold_hours().unwrap();
        assert!((tau - 2.0 / 4.5).abs() < 1e-12);
        // ~26.7 minutes at reported Gemini Pro prices.
        assert!((tau * 60.0 - 26.666).abs() < 0.1);
    }

    #[test]
    fn tau_effective_includes_miss_and_exceeds_tau_hold() {
        let p = PriceSheet::gemini_pro_like();
        let eff = p.tau_effective_hours().unwrap();
        // (2*2.00 - 0.20)/4.50 hours ≈ 50.7 minutes.
        assert!((eff - (2.0 * 2.0 - 0.2) / 4.5).abs() < 1e-12);
        assert!((eff * 60.0 - 50.666).abs() < 0.1);
        assert!(eff > p.tau_hold_hours().unwrap());
    }

    #[test]
    fn regimes() {
        assert_eq!(
            PriceSheet::gemini_pro_like().regime(),
            Regime::StorageMetered
        );
        assert_eq!(
            PriceSheet::anthropic_sonnet_like().regime(),
            Regime::WritePremium
        );
    }
}
