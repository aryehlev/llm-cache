//! Traffic traces: the input to the ROI simulator ([`crate::sim`]).
//!
//! A trace is a time-ordered list of [`Event`]s, each a request reduced to its
//! block-hash path plus token count — **no prompt content** (the same
//! privacy-safe telemetry the engine consumes, so a real trace can be captured
//! from production and replayed without exposing prompts).
//!
//! ## Wire format
//!
//! One event per line, tab-separated: `time_hours \t total_tokens \t h1,h2,...`
//! (block hashes comma-separated; empty block list allowed). Lines beginning
//! with `#` and blank lines are ignored. Parsing is dependency-free and
//! lossless round-trips via [`Event::to_line`] / [`Event::parse_line`].
//!
//! ## Synthetic workloads
//!
//! When you don't yet have a captured trace, the generators here produce the
//! canonical shapes PCOE targets — a business-day office pattern, bursty
//! Poisson arrivals, and a multi-tenant fleet — with a deterministic seeded
//! PRNG so runs are reproducible.

/// One request in a trace.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    /// Arrival time, hours on an arbitrary monotonic clock.
    pub time_hours: f64,
    /// Total input tokens of the request (shared prefix + unique suffix).
    pub total_tokens: u64,
    /// Block-hash path of the prompt (see [`crate::chunk`]).
    pub blocks: Vec<u64>,
}

impl Event {
    /// Serialize to one wire line (no trailing newline).
    pub fn to_line(&self) -> String {
        let mut s = format!("{}\t{}\t", self.time_hours, self.total_tokens);
        for (i, b) in self.blocks.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&b.to_string());
        }
        s
    }

    /// Parse one wire line. Returns `None` for blank/comment lines, `Err` for
    /// malformed data lines.
    pub fn parse_line(line: &str) -> Result<Option<Event>, String> {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }
        let mut fields = line.split('\t');
        let time = fields
            .next()
            .ok_or("missing time")?
            .parse::<f64>()
            .map_err(|e| format!("bad time: {e}"))?;
        let total = fields
            .next()
            .ok_or("missing total_tokens")?
            .parse::<u64>()
            .map_err(|e| format!("bad total_tokens: {e}"))?;
        let blocks_field = fields.next().unwrap_or("");
        let blocks = if blocks_field.is_empty() {
            Vec::new()
        } else {
            blocks_field
                .split(',')
                .map(|h| h.parse::<u64>().map_err(|e| format!("bad block hash: {e}")))
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(Some(Event {
            time_hours: time,
            total_tokens: total,
            blocks,
        }))
    }
}

/// Serialize a whole trace to the wire format (one event per line).
pub fn to_string(events: &[Event]) -> String {
    let mut s = String::from("# pcoe trace v1: time_hours<TAB>total_tokens<TAB>block_hashes\n");
    for e in events {
        s.push_str(&e.to_line());
        s.push('\n');
    }
    s
}

/// Parse a whole trace from the wire format.
pub fn from_str(text: &str) -> Result<Vec<Event>, String> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        match Event::parse_line(line) {
            Ok(Some(e)) => out.push(e),
            Ok(None) => {}
            Err(e) => return Err(format!("line {}: {e}", n + 1)),
        }
    }
    Ok(out)
}

/// Deterministic SplitMix64 PRNG — seeded, reproducible, zero-dependency.
/// Sufficient for synthetic traffic; not cryptographic.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        // 53-bit mantissa precision.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// An exponential inter-arrival gap with the given rate (events/hour).
    pub fn exp_gap(&mut self, rate_per_hour: f64) -> f64 {
        let u = 1.0 - self.unit(); // in (0, 1], avoids ln(0)
        -u.ln() / rate_per_hour.max(1e-9)
    }
}

/// Build a distinct shared prefix of `n_blocks` blocks for tenant/family `id`.
fn prefix_blocks(id: u64, n_blocks: u64) -> Vec<u64> {
    let base = 0x1_0000_0000u64.wrapping_mul(id.wrapping_add(1));
    (0..n_blocks).map(|i| base + i).collect()
}

/// A request = a family's shared prefix + one unique suffix block.
fn make_event(time: f64, prefix: &[u64], suffix_seq: u64, block_tokens: u64) -> Event {
    let mut blocks = prefix.to_vec();
    blocks.push(0xFFFF_0000_0000_0000 ^ suffix_seq);
    Event {
        time_hours: time,
        total_tokens: blocks.len() as u64 * block_tokens,
        blocks,
    }
}

/// Business-hours workload: one shared prefix, Poisson arrivals at
/// `rate_per_hour` during `[open, close)` each day, silence otherwise, for
/// `days` days. This is the canonical PCOE win — a big cache idle every night.
pub fn business_hours(
    prefix_blocks_n: u64,
    rate_per_hour: f64,
    open: f64,
    close: f64,
    days: u64,
    block_tokens: u64,
    seed: u64,
) -> Vec<Event> {
    let prefix = prefix_blocks(0, prefix_blocks_n);
    let mut rng = Rng::new(seed);
    let mut events = Vec::new();
    let mut suffix = 0u64;
    for d in 0..days {
        let day0 = d as f64 * 24.0;
        let mut t = day0 + open;
        loop {
            t += rng.exp_gap(rate_per_hour);
            if t >= day0 + close {
                break;
            }
            events.push(make_event(t, &prefix, suffix, block_tokens));
            suffix += 1;
        }
    }
    events
}

/// Bursty Poisson workload: one shared prefix, arrivals alternating between
/// hot bursts (`hot_rate`) and cold lulls (`cold_rate`), `total_hours` long.
/// Stresses the gap-histogram TTL-tier logic.
pub fn poisson_bursty(
    prefix_blocks_n: u64,
    hot_rate: f64,
    cold_rate: f64,
    burst_hours: f64,
    total_hours: f64,
    block_tokens: u64,
    seed: u64,
) -> Vec<Event> {
    let prefix = prefix_blocks(0, prefix_blocks_n);
    let mut rng = Rng::new(seed);
    let mut events = Vec::new();
    let mut suffix = 0u64;
    let mut t = 0.0;
    let mut hot = true;
    let mut phase_end = burst_hours;
    while t < total_hours {
        let rate = if hot { hot_rate } else { cold_rate };
        t += rng.exp_gap(rate);
        if t >= phase_end {
            hot = !hot;
            phase_end += burst_hours;
            continue;
        }
        if t < total_hours {
            events.push(make_event(t, &prefix, suffix, block_tokens));
            suffix += 1;
        }
    }
    events.sort_by(|a, b| a.time_hours.total_cmp(&b.time_hours));
    events
}

/// Multi-tenant fleet: `tenants` distinct prefixes, each with its own staggered
/// business day, merged into one time-ordered trace. This is the workload no
/// human can hand-tune — the strongest product case.
pub fn multi_tenant(
    tenants: u64,
    prefix_blocks_n: u64,
    rate_per_hour: f64,
    active_hours: f64,
    days: u64,
    block_tokens: u64,
    seed: u64,
) -> Vec<Event> {
    let mut events = Vec::new();
    for tenant in 0..tenants {
        let prefix = prefix_blocks(tenant + 1, prefix_blocks_n);
        // Stagger each tenant's opening hour around the clock.
        let open = (tenant as f64 * 24.0 / tenants as f64) % 24.0;
        let close = open + active_hours;
        let mut rng = Rng::new(seed ^ (tenant.wrapping_mul(0x9E37_79B9)));
        let mut suffix = 0u64;
        for d in 0..days {
            let day0 = d as f64 * 24.0;
            let mut t = day0 + open;
            loop {
                t += rng.exp_gap(rate_per_hour);
                if t >= day0 + close {
                    break;
                }
                events.push(make_event(t, &prefix, suffix, block_tokens));
                suffix += 1;
            }
        }
    }
    events.sort_by(|a, b| a.time_hours.total_cmp(&b.time_hours));
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_round_trips() {
        let e = Event {
            time_hours: 1.25,
            total_tokens: 5120,
            blocks: vec![1, 2, 3],
        };
        let parsed = Event::parse_line(&e.to_line()).unwrap().unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn trace_round_trips_and_skips_comments() {
        let events = vec![
            Event {
                time_hours: 0.0,
                total_tokens: 100,
                blocks: vec![7],
            },
            Event {
                time_hours: 0.5,
                total_tokens: 200,
                blocks: vec![],
            },
        ];
        let text = to_string(&events);
        assert!(text.lines().next().unwrap().starts_with('#'));
        assert_eq!(from_str(&text).unwrap(), events);
    }

    #[test]
    fn business_hours_confined_to_open_window() {
        let ev = business_hours(20, 12.0, 9.0, 17.0, 2, 256, 42);
        assert!(!ev.is_empty());
        for e in &ev {
            let hod = e.time_hours % 24.0;
            assert!((9.0..17.0).contains(&hod), "event at hour-of-day {hod}");
        }
        // Deterministic under a fixed seed.
        assert_eq!(business_hours(20, 12.0, 9.0, 17.0, 2, 256, 42), ev);
    }

    #[test]
    fn multi_tenant_uses_distinct_prefixes() {
        let ev = multi_tenant(3, 10, 20.0, 8.0, 1, 256, 1);
        assert!(ev.len() > 10);
        // Sorted by time.
        for w in ev.windows(2) {
            assert!(w[0].time_hours <= w[1].time_hours);
        }
        // Prefixes differ across tenants (first block distinguishes them).
        let firsts: std::collections::HashSet<u64> = ev
            .iter()
            .filter_map(|e| e.blocks.first().copied())
            .collect();
        assert!(firsts.len() >= 3);
    }

    #[test]
    fn rng_exp_gap_mean_is_reasonable() {
        let mut rng = Rng::new(7);
        let n = 20_000;
        let rate = 5.0;
        let mean: f64 = (0..n).map(|_| rng.exp_gap(rate)).sum::<f64>() / n as f64;
        // E[gap] = 1/rate = 0.2h; allow 5% slack.
        assert!((mean - 0.2).abs() / 0.2 < 0.05, "mean gap {mean}");
    }
}
