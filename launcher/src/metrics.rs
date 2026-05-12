use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tokio::process::Command;

use crate::proxy::stats::SharedStats;
use crate::recipe::{Model, Stack};

const HISTORY_CAP: usize = 120; // ~4 minutes at 2s/sample

#[derive(Debug, Default, Clone)]
pub struct ModelState {
    pub healthy: bool,
    pub last_seen: Option<Instant>,
    pub ready_since: Option<Instant>,
    pub num_requests_running: f64,
    pub num_requests_waiting: f64,
    pub gpu_cache_usage_perc: f64,
    pub prompt_tokens_total: f64,
    pub generation_tokens_total: f64,
    pub prompt_rate: f64,
    pub gen_rate: f64,
    /// Rolling history of recent samples for sparklines (oldest → newest).
    pub history: VecDeque<Sample>,
    last_scrape_at: Option<Instant>,
    last_prompt: Option<f64>,
    last_gen: Option<f64>,
}

#[derive(Debug, Default, Clone)]
pub struct Sample {
    pub prompt_rate: f64,
    pub gen_rate: f64,
    pub running: f64,
    pub kv_pct: f64, // 0..1
}

#[derive(Debug, Default, Clone)]
pub struct HostState {
    pub gpu_util: Option<f64>,
    pub mem_used_gb: Option<f64>,
    pub mem_total_gb: Option<f64>,
    pub last_seen: Option<Instant>,
}

#[derive(Debug, Default, Clone)]
pub struct ProxyState {
    pub running: bool,
    pub uptime: Option<Duration>,
    pub started_at: Option<Instant>,
    pub requests_total: u64,
    pub requests_by_status: HashMap<u16, u64>,
    pub requests_by_route: HashMap<String, u64>,
    pub last_request_at: Option<Instant>,
}

#[derive(Debug, Default)]
pub struct AppState {
    pub models: HashMap<String, ModelState>,
    pub host: HostState,
    pub proxy: ProxyState,
    pub chats: HashMap<String, ChatState>,
}

#[derive(Debug, Clone)]
pub struct ChatState {
    pub messages: Vec<ChatMessage>,
    pub input: String,
    pub in_flight: bool,
    pub error: Option<String>,
    /// Lines scrolled past the top of the message viewport.
    pub scroll: u16,
    /// When true, the view auto-tracks the bottom as new content arrives.
    /// Set false when the user scrolls up; re-set true when they scroll back
    /// to the bottom or submit a new message.
    pub auto_follow: bool,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            in_flight: false,
            error: None,
            scroll: 0,
            auto_follow: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChatMessage {
    pub role: String,    // "user" | "assistant"
    pub content: String,
    /// Accumulated reasoning/thinking trace (empty for non-assistant messages).
    pub thinking: String,
    /// Flips to true once the model emits its first non-thinking content
    /// delta — at that point the thinking block stops animating and collapses
    /// to a finished summary.
    pub thinking_complete: bool,
    /// Wall-clock anchors for displaying "thought for Ns".
    pub thinking_started_at: Option<Instant>,
    pub thinking_ended_at: Option<Instant>,
}

pub type SharedState = Arc<RwLock<AppState>>;

pub fn new_state() -> SharedState { Arc::new(RwLock::new(AppState::default())) }

pub async fn spawn_pollers(stack: Arc<Stack>, state: SharedState, proxy_stats: SharedStats) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client");
    for m in &stack.models {
        let st = state.clone();
        let host = stack.host.clone();
        let model = m.clone();
        let c = client.clone();
        tokio::spawn(async move { poll_vllm(c, host, model, st).await; });
    }
    {
        let st = state.clone();
        let host = stack.host.clone();
        tokio::spawn(async move { poll_host(host, st).await; });
    }
    {
        let st = state.clone();
        let stats = proxy_stats.clone();
        tokio::spawn(async move { mirror_proxy_stats(stats, st).await; });
    }
}

async fn poll_vllm(client: reqwest::Client, host: String, model: Model, state: SharedState) {
    let metrics_url = format!("http://{}:{}/metrics", host, model.port);
    let health_url = format!("http://{}:{}/v1/models", host, model.port);
    loop {
        let healthy = client.get(&health_url).send().await
            .map(|r| r.status().is_success()).unwrap_or(false);
        let metrics = if let Ok(r) = client.get(&metrics_url).send().await {
            if r.status().is_success() { r.text().await.ok() } else { None }
        } else { None };
        {
            let mut s = state.write();
            let entry = s.models.entry(model.name.clone()).or_default();
            entry.healthy = healthy;
            if healthy {
                let now = Instant::now();
                entry.last_seen = Some(now);
                if entry.ready_since.is_none() { entry.ready_since = Some(now); }
            } else {
                entry.ready_since = None;
            }
            if let Some(text) = metrics {
                let parsed = parse_vllm_metrics(&text);
                entry.num_requests_running = parsed.running;
                entry.num_requests_waiting = parsed.waiting;
                entry.gpu_cache_usage_perc = parsed.kv_pct;
                let now = Instant::now();
                if let (Some(prev_t), Some(prev_p), Some(prev_g)) =
                    (entry.last_scrape_at, entry.last_prompt, entry.last_gen) {
                    let dt = now.duration_since(prev_t).as_secs_f64().max(1e-3);
                    entry.prompt_rate = ((parsed.prompt_total - prev_p).max(0.0)) / dt;
                    entry.gen_rate = ((parsed.gen_total - prev_g).max(0.0)) / dt;
                }
                entry.prompt_tokens_total = parsed.prompt_total;
                entry.generation_tokens_total = parsed.gen_total;
                entry.last_scrape_at = Some(now);
                entry.last_prompt = Some(parsed.prompt_total);
                entry.last_gen = Some(parsed.gen_total);
                // Push a history sample for the sparkline graphs.
                entry.history.push_back(Sample {
                    prompt_rate: entry.prompt_rate,
                    gen_rate: entry.gen_rate,
                    running: entry.num_requests_running,
                    kv_pct: entry.gpu_cache_usage_perc,
                });
                while entry.history.len() > HISTORY_CAP { entry.history.pop_front(); }
            }
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

#[derive(Default)]
struct VllmSnapshot {
    running: f64,
    waiting: f64,
    kv_pct: f64,
    prompt_total: f64,
    gen_total: f64,
}

fn parse_vllm_metrics(text: &str) -> VllmSnapshot {
    let mut s = VllmSnapshot::default();
    for line in text.lines() {
        if line.starts_with('#') { continue; }
        let (name_part, value_part) = match line.rsplit_once(' ') {
            Some((a, b)) => (a, b),
            None => continue,
        };
        let name = name_part.split('{').next().unwrap_or("").trim();
        let value: f64 = match value_part.parse() { Ok(v) => v, Err(_) => continue };
        match name {
            "vllm:num_requests_running" => s.running = value,
            "vllm:num_requests_waiting" => s.waiting = value,
            "vllm:kv_cache_usage_perc" | "vllm:gpu_cache_usage_perc" => s.kv_pct = value,
            "vllm:prompt_tokens_total" => s.prompt_total += value,
            "vllm:generation_tokens_total" => s.gen_total += value,
            _ => {}
        }
    }
    s
}

async fn poll_host(host: String, state: SharedState) {
    let script = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null; echo MEM:; awk '/MemTotal|MemAvailable/ {print $1,$2}' /proc/meminfo";
    loop {
        let out = Command::new("ssh")
            .arg("-o").arg("BatchMode=yes")
            .arg("-o").arg("ConnectTimeout=3")
            .arg(&host).arg(script)
            .output().await;
        if let Ok(o) = out {
            if o.status.success() {
                let text = String::from_utf8_lossy(&o.stdout);
                let (util, used_gb, total_gb) = parse_host(&text);
                let mut s = state.write();
                s.host.gpu_util = util;
                s.host.mem_used_gb = used_gb;
                s.host.mem_total_gb = total_gb;
                s.host.last_seen = Some(Instant::now());
            }
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

fn parse_host(text: &str) -> (Option<f64>, Option<f64>, Option<f64>) {
    let mut util = None;
    let mut mem_total_kb: Option<f64> = None;
    let mut mem_avail_kb: Option<f64> = None;
    let mut past_marker = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        if line == "MEM:" { past_marker = true; continue; }
        if !past_marker {
            if let Ok(v) = line.parse::<f64>() { util = Some(v); }
        } else {
            let mut it = line.split_whitespace();
            let key = it.next().unwrap_or("");
            let val = it.next().and_then(|v| v.parse::<f64>().ok());
            match key {
                "MemTotal:" => mem_total_kb = val,
                "MemAvailable:" => mem_avail_kb = val,
                _ => {}
            }
        }
    }
    let used_gb = match (mem_total_kb, mem_avail_kb) {
        (Some(t), Some(a)) => Some((t - a) / 1024.0 / 1024.0),
        _ => None,
    };
    let total_gb = mem_total_kb.map(|t| t / 1024.0 / 1024.0);
    (util, used_gb, total_gb)
}

async fn mirror_proxy_stats(stats: SharedStats, state: SharedState) {
    loop {
        {
            let s = stats.read();
            let mut app = state.write();
            app.proxy.running = s.started_at.is_some();
            app.proxy.started_at = s.started_at;
            app.proxy.uptime = s.started_at.map(|t| t.elapsed());
            app.proxy.requests_total = s.requests_total;
            app.proxy.requests_by_status = s.requests_by_status.clone();
            app.proxy.requests_by_route = s.requests_by_route.clone();
            app.proxy.last_request_at = s.last_request_at;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}
