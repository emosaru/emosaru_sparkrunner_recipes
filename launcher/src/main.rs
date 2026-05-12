use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

mod chat;
mod metrics;
mod proxy;
mod recipe;
mod sparkrun;
mod tui;

use recipe::Stack;

#[derive(Parser, Debug)]
#[command(name = "stackctl", version, about = "Run, stop, and monitor a sparkrun model stack with an embedded OpenAI+Anthropic proxy.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Launch each model, start the embedded proxy on both ports, then drop into the TUI.
    Up {
        stack: PathBuf,
        /// Skip the TUI after launch; the proxy keeps running until you interrupt.
        #[arg(long)]
        no_tui: bool,
        /// Skip `sparkrun run` calls; only start the proxy. Useful for testing or when models are already up.
        #[arg(long)]
        no_launch: bool,
    },
    /// Stop each model on the upstream host.
    Down { stack: PathBuf },
    /// Attach the TUI to a running stack (read-only — proxy must be running in another stackctl process to see request stats).
    Tui { stack: PathBuf },
    /// One-shot status print.
    Status { stack: PathBuf },
    /// Print the resolved route map (model name → upstream).
    Routes { stack: PathBuf },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Up { stack, no_tui, no_launch } => cmd_up(stack, no_tui, no_launch).await,
        Cmd::Down { stack } => cmd_down(stack).await,
        Cmd::Tui { stack } => cmd_tui(stack).await,
        Cmd::Status { stack } => cmd_status(stack).await,
        Cmd::Routes { stack } => cmd_routes(stack),
    }
}

async fn cmd_up(stack_path: PathBuf, no_tui: bool, no_launch: bool) -> Result<()> {
    let stack = Arc::new(Stack::load(&stack_path)?);
    println!("▶ stack: {} ({} models)", stack.name, stack.models.len());

    if !no_launch {
        sparkrun::ensure_sparkrun_on_path().await?;
        for model in &stack.models {
            println!("\n▶ [{}] launching on :{}…", model.name, model.port);
            sparkrun::run_model(&stack, model).await
                .with_context(|| format!("launch {}", model.name))?;
            print!("  waiting for :{} ", model.port);
            use std::io::Write;
            std::io::stdout().flush().ok();
            sparkrun::wait_ready(&stack.host, model.port, Duration::from_secs(model.ready_timeout_seconds)).await
                .with_context(|| format!("{} did not become ready", model.name))?;
            println!("ready");
        }
    } else {
        println!("  (--no-launch: skipping sparkrun run)");
    }

    println!("\n▶ starting embedded proxy…");
    let handle = proxy::start(stack.clone()).await?;
    println!(
        "  OpenAI    http://{}:{}",
        stack.proxy.host, stack.proxy.openai_port
    );
    println!(
        "  Anthropic http://{}:{}",
        stack.proxy.host, stack.proxy.anthropic_port
    );
    println!("  routes: {}", handle.routes.len());
    println!("✓ stack up.");

    let state = metrics::new_state();
    let host_pollers = stack.clone();
    metrics::spawn_pollers(host_pollers.clone(), state.clone(), handle.stats.clone()).await;

    if no_tui {
        println!("  (--no-tui set; press Ctrl-C to stop the proxy)");
        tokio::signal::ctrl_c().await.ok();
        proxy::stop(&handle);
        return Ok(());
    }
    tui::run(stack, state).await?;
    proxy::stop(&handle);
    Ok(())
}

async fn cmd_down(stack_path: PathBuf) -> Result<()> {
    let stack = Stack::load(&stack_path)?;
    println!("▶ stopping models…");
    for model in stack.models.iter().rev() {
        println!("  • {}", model.name);
        if let Err(e) = sparkrun::stop_model(&stack, model).await {
            eprintln!("    error: {e}");
        }
    }
    println!("\n✓ stack down. verify: sparkrun status");
    Ok(())
}

async fn cmd_tui(stack_path: PathBuf) -> Result<()> {
    let stack = Arc::new(Stack::load(&stack_path)?);
    let state = metrics::new_state();
    // No proxy stats source available in attach mode; only model + host metrics will populate.
    let dummy_stats = proxy::stats::new();
    metrics::spawn_pollers(stack.clone(), state.clone(), dummy_stats).await;
    tui::run(stack, state).await
}

async fn cmd_status(stack_path: PathBuf) -> Result<()> {
    let stack = Stack::load(&stack_path)?;
    let state = metrics::new_state();
    let st = Arc::new(stack.clone());
    let dummy_stats = proxy::stats::new();
    metrics::spawn_pollers(st.clone(), state.clone(), dummy_stats).await;
    tokio::time::sleep(Duration::from_secs(3)).await;
    let s = state.read();
    println!("stack: {}", stack.name);
    println!("  host {}: util={:?} mem={:?}/{:?} GB",
             stack.host, s.host.gpu_util, s.host.mem_used_gb, s.host.mem_total_gb);
    for m in &stack.models {
        let ms = s.models.get(&m.name).cloned().unwrap_or_default();
        println!("  • {} :{}  healthy={}  running={}  waiting={}  kv={:.1}%  prompt={:.1}/s  decode={:.1}/s",
                 m.name, m.port, ms.healthy, ms.num_requests_running as u64, ms.num_requests_waiting as u64,
                 ms.gpu_cache_usage_perc * 100.0, ms.prompt_rate, ms.gen_rate);
    }
    Ok(())
}

fn cmd_routes(stack_path: PathBuf) -> Result<()> {
    let stack = Stack::load(&stack_path)?;
    let routes = proxy::router::build(&stack);
    let mut entries: Vec<(&String, &proxy::router::UpstreamRoute)> = routes.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (name, r) in entries {
        println!("{:50}  →  {}  ({})", name, r.api_base, r.canonical);
    }
    Ok(())
}
