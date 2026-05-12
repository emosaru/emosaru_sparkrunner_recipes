use std::collections::HashMap;
use std::sync::Arc;

use crate::recipe::Stack;

#[derive(Debug, Clone)]
pub struct UpstreamRoute {
    pub canonical: String,    // upstream model identifier (sent in payload)
    pub api_base: String,     // http://host:port/v1
}

pub type RouteMap = Arc<HashMap<String, UpstreamRoute>>;

pub fn build(stack: &Stack) -> RouteMap {
    let mut m = HashMap::new();
    for model in &stack.models {
        let route = UpstreamRoute {
            canonical: model.name.clone(),
            api_base: format!("http://{}:{}/v1", stack.host, model.port),
        };
        // Original (e.g. "Qwen/Qwen3.6-35B-A3B-FP8") — used by OpenAI clients.
        m.insert(model.name.clone(), route.clone());
        // claude-prefixed, slash-sanitized (e.g. "claude-Qwen-Qwen3.6-35B-A3B-FP8")
        // — what Claude Code's /model picker filters on; real Anthropic IDs
        // never contain `/`, so we replace it with `-` and back-translate by
        // registering this form as a first-class route key.
        let pp = picker_id(&model.name);
        if pp != model.name {
            m.insert(pp, route.clone());
        }
        for alias in &model.aliases {
            m.insert(alias.clone(), route.clone());
        }
    }
    Arc::new(m)
}

/// Generate a Claude-Code-picker-compatible model id from a canonical name.
/// `claude-` prefix is added if missing; `/` is replaced with `-` so the
/// id matches Anthropic's id grammar.
pub fn picker_id(name: &str) -> String {
    let sanitized = name.replace('/', "-");
    if sanitized.starts_with("claude-") || sanitized.starts_with("anthropic") {
        sanitized
    } else {
        format!("claude-{sanitized}")
    }
}

pub fn lookup<'a>(routes: &'a RouteMap, name: &str) -> Option<&'a UpstreamRoute> {
    routes.get(name)
}
