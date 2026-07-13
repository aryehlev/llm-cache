//! End-to-end simulation of DESIGN.md §4.1: a document-QA assistant with a
//! 200K-token shared prefix, 12 requests/hour for 8 business hours, silence
//! overnight, on Gemini-Pro-like storage-metered prices.
//!
//! The design doc's machine-checked totals for this workload:
//!   no caching        ≈ $38.40/day
//!   cache-everything  ≈ $25.84/day
//!   PCOE              ≈ $11.84/day (≈ −69%)
//! The simulation must land near the PCOE figure (slightly above, since the
//! first few requests run uncached while the rate estimator warms up).

use llm_cache::{Action, Config, Engine, PriceSheet};

#[test]
fn business_day_costs_match_design_doc() {
    let prices = PriceSheet::gemini_pro_like();
    let p_in = prices.input_per_mtok;
    let p_read = prices.cached_read_per_mtok;
    let p_store = prices.storage_per_mtok_hour.unwrap();
    let tau_hold = prices.tau_hold_hours().unwrap();

    let mut eng = Engine::new(prices.clone(), Config::default());

    // 782 blocks x 256 tokens = 200_192 tokens of shared prefix.
    let prefix: Vec<u64> = (0..782u64).map(|i| 0x9000_0000 + i).collect();
    let prefix_tokens: u64 = 782 * 256;
    let mtok = prefix_tokens as f64 / 1e6;

    let mut cost = 0.0;
    let mut cached_reads = 0u32;
    let mut creates = 0u32;
    let mut deletes = 0u32;
    let mut storage_open: Option<(f64, u64)> = None;
    let mut delete_time: Option<f64> = None;

    let handle_action = |a: &Action,
                         now: f64,
                         cost: &mut f64,
                         creates: &mut u32,
                         deletes: &mut u32,
                         storage_open: &mut Option<(f64, u64)>,
                         delete_time: &mut Option<f64>| {
        match *a {
            Action::Create { tokens, .. } => {
                *creates += 1;
                *cost += tokens as f64 / 1e6 * p_in; // creation bills tokens once
                *storage_open = Some((now, tokens));
            }
            Action::Delete { .. } => {
                *deletes += 1;
                if let Some((t0, tokens)) = storage_open.take() {
                    *cost += (now - t0) * (tokens as f64 / 1e6) * p_store;
                }
                *delete_time = Some(now);
            }
            Action::Extend { .. } => {} // storage accrues create -> delete
        }
    };

    // 10 hours in 1-minute ticks; a request every 5 minutes during the first 8.
    for m in 0..=600u32 {
        let now = m as f64 / 60.0;
        if m % 5 == 0 && m < 480 {
            let mut blocks = prefix.clone();
            blocks.push(0xdead_beef + m as u64); // unique per-request suffix
            let obs = eng.observe(&blocks, now);
            let ct = obs.cached_tokens.min(prefix_tokens);
            if ct > 0 {
                cached_reads += 1;
            }
            cost += ct as f64 / 1e6 * p_read + (prefix_tokens - ct) as f64 / 1e6 * p_in;
            for a in &obs.actions {
                if let Action::Create { node, .. } = a {
                    eng.confirm_create(*node, now); // instantly-successful adapter
                }
                handle_action(
                    a,
                    now,
                    &mut cost,
                    &mut creates,
                    &mut deletes,
                    &mut storage_open,
                    &mut delete_time,
                );
            }
        }
        for a in eng.tick(now) {
            handle_action(
                &a,
                now,
                &mut cost,
                &mut creates,
                &mut deletes,
                &mut storage_open,
                &mut delete_time,
            );
        }
    }

    assert_eq!(creates, 1, "one cache creation for the day");
    assert_eq!(deletes, 1, "one end-of-day deletion");
    assert!(cached_reads >= 90, "cached reads: {cached_reads}");

    // Ski-rental exit: deletion lands within a tick of tau_hold after the last
    // hit (last request at minute 475).
    let last_hit = 475.0 / 60.0;
    let dt = delete_time.unwrap();
    assert!(
        dt >= last_hit + tau_hold - 1e-9 && dt <= last_hit + tau_hold + 2.0 / 60.0,
        "deleted at {dt:.4}h, expected ~{:.4}h",
        last_hit + tau_hold
    );

    // Cost comparison against the design doc's reference policies.
    let no_cache = 96.0 * mtok * p_in; // ≈ $38.44
    let hold_24h = mtok * p_in + 96.0 * mtok * p_read + 24.0 * mtok * p_store; // ≈ $25.86
    assert!(
        cost < 14.0,
        "PCOE day cost ${cost:.2} should be near the doc's $11.84 (+ warmup)"
    );
    assert!(
        cost < hold_24h,
        "must beat cache-everything (${hold_24h:.2})"
    );
    assert!(
        cost < 0.4 * no_cache,
        "must save >60% vs no caching (${no_cache:.2})"
    );
}
