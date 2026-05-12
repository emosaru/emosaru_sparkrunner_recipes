use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Stack {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub host: String,
    pub proxy: ProxyCfg,
    pub models: Vec<Model>,
    #[serde(skip)]
    pub source_path: PathBuf,
    #[serde(skip)]
    pub repo_root: PathBuf,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProxyCfg {
    #[serde(default = "default_proxy_host")]
    pub host: String,
    #[serde(default = "default_openai_port")]
    pub openai_port: u16,
    #[serde(default = "default_anthropic_port")]
    pub anthropic_port: u16,
}

fn default_proxy_host() -> String { "0.0.0.0".into() }
fn default_openai_port() -> u16 { 4000 }
fn default_anthropic_port() -> u16 { 4001 }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Model {
    pub recipe: String,
    pub name: String,
    pub port: u16,
    #[serde(default = "default_ready_timeout")]
    pub ready_timeout_seconds: u64,
    #[serde(default)]
    pub aliases: Vec<String>,
}

fn default_ready_timeout() -> u64 { 1800 }

impl Stack {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read stack file {}", path.display()))?;
        let mut stack: Stack = serde_yaml::from_str(&raw)
            .with_context(|| format!("parse stack yaml {}", path.display()))?;
        stack.source_path = path
            .canonicalize()
            .with_context(|| format!("canonicalize {}", path.display()))?;
        stack.repo_root = stack
            .source_path
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(stack)
    }

    pub fn recipe_path(&self, model: &Model) -> PathBuf {
        let p = PathBuf::from(&model.recipe);
        if p.is_absolute() { p } else { self.repo_root.join(p) }
    }

}
