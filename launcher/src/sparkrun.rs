use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use tokio::process::Command;

use crate::recipe::{Model, Stack};

pub async fn run_model(stack: &Stack, model: &Model) -> Result<()> {
    let recipe = stack.recipe_path(model);
    if !recipe.exists() {
        bail!("recipe not found: {}", recipe.display());
    }
    let status = Command::new("sparkrun")
        .arg("run")
        .arg(&recipe)
        .arg("--ensure")
        .arg("--no-follow")
        .status()
        .await
        .with_context(|| "spawn sparkrun run")?;
    if !status.success() {
        bail!("sparkrun run failed for {}", recipe.display());
    }
    Ok(())
}

pub async fn stop_model(stack: &Stack, model: &Model) -> Result<()> {
    let recipe = stack.recipe_path(model);
    let status = Command::new("sparkrun")
        .arg("stop")
        .arg(&recipe)
        .status()
        .await
        .with_context(|| "spawn sparkrun stop")?;
    if !status.success() {
        // sparkrun stop returns non-zero if not running; treat as soft failure.
        eprintln!("  (sparkrun stop returned non-zero for {})", recipe.display());
    }
    Ok(())
}

pub async fn wait_ready(host: &str, port: u16, timeout: Duration) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()?;
    let url = format!("http://{host}:{port}/v1/models");
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(r) = client.get(&url).send().await {
            if r.status().is_success() {
                return Ok(());
            }
        }
        if Instant::now() >= deadline {
            return Err(anyhow!("{} never reported ready within {:?}", url, timeout));
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

pub async fn ensure_sparkrun_on_path() -> Result<()> {
    Command::new("sparkrun")
        .arg("--version")
        .output()
        .await
        .map(|_| ())
        .map_err(|e| anyhow!("sparkrun CLI not found on PATH: {e}"))
}
