use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;

#[derive(Debug, Default, Clone)]
pub struct ProxyStats {
    pub requests_total: u64,
    pub requests_by_status: HashMap<u16, u64>,
    pub requests_by_route: HashMap<String, u64>, // "openai:chat" | "anthropic:messages" | ...
    pub last_request_at: Option<Instant>,
    pub started_at: Option<Instant>,
}

pub type SharedStats = Arc<RwLock<ProxyStats>>;

pub fn new() -> SharedStats {
    Arc::new(RwLock::new(ProxyStats::default()))
}

pub fn record(stats: &SharedStats, route: &str, status: u16) {
    let mut s = stats.write();
    s.requests_total += 1;
    *s.requests_by_status.entry(status).or_insert(0) += 1;
    *s.requests_by_route.entry(route.to_string()).or_insert(0) += 1;
    s.last_request_at = Some(Instant::now());
}
