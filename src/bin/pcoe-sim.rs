//! `pcoe-sim` — the ROI command line for PCOE.
//!
//! Generate a synthetic traffic trace, or replay a captured one, and print what
//! each caching policy would have cost — so the savings figure is *your*
//! traffic's, not a synthetic claim.
//!
//! ```text
//! pcoe-sim gen <scenario> [--out FILE] [--seed N] [--days N]
//! pcoe-sim run (--trace FILE | --scenario NAME) [--provider gemini] [--seed N] [--days N]
//! pcoe-sim scenarios
//! ```
//!
//! Scenarios: `business-day`, `bursty`, `multi-tenant`.

use std::process::ExitCode;

use llm_cache::price::PriceSheet;
use llm_cache::sim::{compare, Comparison, Policy};
use llm_cache::trace::{self, Event};
use llm_cache::Config;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    }
    let result = match args[0].as_str() {
        "gen" => cmd_gen(&args[1..]),
        "run" => cmd_run(&args[1..]),
        "scenarios" => {
            print_scenarios();
            Ok(())
        }
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "\
pcoe-sim — measure what PCOE would save on a traffic trace

USAGE:
    pcoe-sim gen <scenario> [--out FILE] [--seed N] [--days N]
    pcoe-sim run (--trace FILE | --scenario NAME) [--provider gemini] [--seed N] [--days N]
    pcoe-sim scenarios

SCENARIOS:
    business-day    one 200K-token prefix, 8h/day, silent overnight
    bursty          one prefix, alternating hot/cold Poisson arrivals
    multi-tenant    a fleet of prefixes with staggered business hours

OPTIONS:
    --trace FILE     replay a captured trace (tab-separated; see docs)
    --scenario NAME  generate and replay a synthetic scenario
    --out FILE       (gen) write the trace here instead of stdout
    --provider NAME  gemini (default) — storage-metered regime
    --seed N         PRNG seed for synthetic generation (default 42)
    --days N         number of days to simulate (default 7)";

struct Opts {
    trace: Option<String>,
    scenario: Option<String>,
    out: Option<String>,
    provider: String,
    seed: u64,
    days: u64,
}

fn parse_opts(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        trace: None,
        scenario: None,
        out: None,
        provider: "gemini".into(),
        seed: 42,
        days: 7,
    };
    let mut i = 0;
    // A bare first positional is a scenario name (for `gen <scenario>`).
    if i < args.len() && !args[i].starts_with("--") {
        o.scenario = Some(args[i].clone());
        i += 1;
    }
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> {
            args.get(i + 1)
                .ok_or_else(|| format!("{} requires a value", args[i]))
        };
        match args[i].as_str() {
            "--trace" => {
                o.trace = Some(need(i)?.clone());
                i += 2;
            }
            "--scenario" => {
                o.scenario = Some(need(i)?.clone());
                i += 2;
            }
            "--out" => {
                o.out = Some(need(i)?.clone());
                i += 2;
            }
            "--provider" => {
                o.provider = need(i)?.clone();
                i += 2;
            }
            "--seed" => {
                o.seed = need(i)?.parse().map_err(|e| format!("bad --seed: {e}"))?;
                i += 2;
            }
            "--days" => {
                o.days = need(i)?.parse().map_err(|e| format!("bad --days: {e}"))?;
                i += 2;
            }
            other => return Err(format!("unknown option '{other}'")),
        }
    }
    Ok(o)
}

fn price_sheet(provider: &str) -> Result<PriceSheet, String> {
    match provider {
        "gemini" => Ok(PriceSheet::gemini_pro_like()),
        other => Err(format!(
            "unknown provider '{other}' (only 'gemini' has a storage-metered \
             simulation; Anthropic caching is write-premium — use the library's \
             plan() API)"
        )),
    }
}

fn generate(scenario: &str, seed: u64, days: u64) -> Result<Vec<Event>, String> {
    let block_tokens = 256;
    match scenario {
        // ~200K-token prefix, 12 req/h, 9-17h.
        "business-day" => Ok(trace::business_hours(
            782,
            12.0,
            9.0,
            17.0,
            days,
            block_tokens,
            seed,
        )),
        // ~100K-token prefix, hot 60/h vs cold 2/h in 1h phases.
        "bursty" => Ok(trace::poisson_bursty(
            390,
            60.0,
            2.0,
            1.0,
            days as f64 * 24.0,
            block_tokens,
            seed,
        )),
        // 12 tenants, ~100K prefixes, staggered 8h days.
        "multi-tenant" => Ok(trace::multi_tenant(
            12,
            390,
            15.0,
            8.0,
            days,
            block_tokens,
            seed,
        )),
        other => Err(format!(
            "unknown scenario '{other}' (see `pcoe-sim scenarios`)"
        )),
    }
}

fn print_scenarios() {
    println!("Available synthetic scenarios (for --scenario or `gen`):\n");
    println!("  business-day   one ~200K-token prefix, 12 req/h during 9-17h,");
    println!("                 silent overnight. The canonical PCOE win.");
    println!("  bursty         one ~100K-token prefix, alternating hot (60/h)");
    println!("                 and cold (2/h) Poisson phases. Stresses TTL logic.");
    println!("  multi-tenant   12 tenants, ~100K prefixes each, business hours");
    println!("                 staggered around the clock. Un-hand-tunable.");
}

fn cmd_gen(args: &[String]) -> Result<(), String> {
    let o = parse_opts(args)?;
    let scenario = o
        .scenario
        .ok_or("gen requires a scenario name (see `pcoe-sim scenarios`)")?;
    let events = generate(&scenario, o.seed, o.days)?;
    let text = trace::to_string(&events);
    match &o.out {
        Some(path) => {
            std::fs::write(path, &text).map_err(|e| format!("writing {path}: {e}"))?;
            eprintln!("wrote {} events to {path}", events.len());
        }
        None => print!("{text}"),
    }
    Ok(())
}

fn cmd_run(args: &[String]) -> Result<(), String> {
    let o = parse_opts(args)?;
    let prices = price_sheet(&o.provider)?;

    let (events, label) = match (&o.trace, &o.scenario) {
        (Some(path), _) => {
            let text = std::fs::read_to_string(path).map_err(|e| format!("reading {path}: {e}"))?;
            (trace::from_str(&text)?, format!("trace {path}"))
        }
        (None, Some(scenario)) => (
            generate(scenario, o.seed, o.days)?,
            format!("scenario {scenario} ({} days, seed {})", o.days, o.seed),
        ),
        (None, None) => return Err("run requires --trace FILE or --scenario NAME".into()),
    };

    if events.is_empty() {
        return Err("trace is empty".into());
    }

    let cfg = Config::default();
    let cmp = compare(&events, &prices, &cfg);
    print_report(&cmp, &label, events.len(), o.provider.as_str());
    Ok(())
}

/// Format a savings percentage: positive = cheaper (`-69%`), negative = more
/// expensive (`+8%`), so a policy that costs *more* than the baseline reads
/// clearly instead of rendering a double-minus.
fn fmt_savings(pct: f64) -> String {
    if pct >= 0.0 {
        format!("-{:.0}%", pct)
    } else {
        format!("+{:.0}%", -pct)
    }
}

fn policy_name(p: Policy) -> String {
    match p {
        Policy::NoCache => "no caching".into(),
        Policy::CacheEverything => "cache-everything".into(),
        Policy::StaticTtl(h) => format!("static TTL ({:.0}m idle)", h * 60.0),
        Policy::Pcoe => "PCOE".into(),
        Policy::ReferenceOracle => "offline oracle".into(),
    }
}

fn print_report(cmp: &Comparison, label: &str, n_events: usize, provider: &str) {
    let no_cache = cmp.get(Policy::NoCache).unwrap();
    let span_hours = 0.0; // filled below if desired; kept simple

    println!("\n  PCOE ROI report — {label}");
    println!("  provider: {provider} (storage-metered)   requests: {n_events}");
    let _ = span_hours;
    println!();
    println!(
        "  {:<26} {:>12} {:>10} {:>9} {:>10}",
        "policy", "cost ($)", "vs no-$", "hit rate", "ops(c/e/d)"
    );
    println!("  {}", "-".repeat(70));
    for (p, r) in &cmp.reports {
        let vs = if *p == Policy::NoCache {
            "—".to_string()
        } else {
            fmt_savings(r.savings_vs(no_cache))
        };
        let hit = if *p == Policy::ReferenceOracle {
            "—".to_string()
        } else {
            format!("{:.0}%", r.hit_rate() * 100.0)
        };
        let ops = if *p == Policy::NoCache || *p == Policy::ReferenceOracle {
            "—".to_string()
        } else {
            format!("{}/{}/{}", r.creates, r.extends, r.deletes)
        };
        let mark = if *p == Policy::Pcoe { "▶ " } else { "  " };
        println!(
            "{}{:<26} {:>12.2} {:>10} {:>9} {:>10}",
            mark,
            policy_name(*p),
            r.total_cost,
            vs,
            hit,
            ops
        );
    }
    println!();

    let pcoe = cmp.get(Policy::Pcoe).unwrap();
    let every = cmp.get(Policy::CacheEverything).unwrap();
    println!(
        "  PCOE vs no caching:       {:>5}   (${:.2} -> ${:.2})",
        fmt_savings(pcoe.savings_vs(no_cache)),
        no_cache.total_cost,
        pcoe.total_cost
    );
    println!(
        "  PCOE vs cache-everything: {:>5}   (${:.2} -> ${:.2})",
        fmt_savings(pcoe.savings_vs(every)),
        every.total_cost,
        pcoe.total_cost
    );
    if let Some(ratio) = cmp.competitive_ratio() {
        println!(
            "  PCOE vs offline optimal:  {:.2}x  (1.00x = perfect hindsight)",
            ratio
        );
    }
    println!();
}
