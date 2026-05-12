// Proxy module: in-process axum server that exposes two ports.
//   - openai_port: OpenAI-compatible (/v1/models, /v1/chat/completions)
//   - anthropic_port: Anthropic-compatible (/v1/models, /v1/messages),
//     with full Anthropic↔OpenAI translation (incl. streaming SSE).
//
// vLLM upstreams stay OpenAI-shaped; the Anthropic-side server translates
// requests and responses on the fly. Request stats land in shared AppState
// so the TUI can render them directly (no log tailing).

pub mod anthropic;
pub mod log;
pub mod openai;
pub mod router;
pub mod server;
pub mod sse;
pub mod stats;
pub mod translate;
pub mod types;

pub use server::{start, stop};
