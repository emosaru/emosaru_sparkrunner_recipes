// Wire-format types for OpenAI and Anthropic APIs.
//
// We deliberately accept "unknown" fields with #[serde(flatten)] extras on a
// few hot paths so we forward client-set knobs (e.g., reasoning_effort,
// chat_template_kwargs) without enumerating every vLLM-specific option.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------- OpenAI ----

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>, // string OR array of parts
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    // vLLM emits the thinking trace as either `reasoning_content` (older) or
    // `reasoning` (newer). Accept both on the wire, normalize to one field.
    #[serde(default, alias = "reasoning", skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: OpenAIFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: OpenAIToolCallFn,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIToolCallFn {
    pub name: String,
    pub arguments: String, // JSON-encoded string
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIChatResponse {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    #[serde(default)]
    pub usage: Option<OpenAIUsage>,
    #[serde(default)]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIChoice {
    pub index: u32,
    pub message: OpenAIMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct OpenAIUsage {
    #[serde(default)] pub prompt_tokens: u32,
    #[serde(default)] pub completion_tokens: u32,
    #[serde(default)] pub total_tokens: u32,
}

// --- Streaming delta event from upstream vLLM ---------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIStreamEvent {
    #[serde(default)] pub id: Option<String>,
    #[serde(default)] pub object: Option<String>,
    #[serde(default)] pub created: Option<u64>,
    #[serde(default)] pub model: Option<String>,
    #[serde(default)] pub choices: Vec<OpenAIStreamChoice>,
    #[serde(default)] pub usage: Option<OpenAIUsage>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIStreamChoice {
    pub index: u32,
    #[serde(default)] pub delta: OpenAIStreamDelta,
    #[serde(default)] pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct OpenAIStreamDelta {
    #[serde(default)] pub role: Option<String>,
    #[serde(default)] pub content: Option<String>,
    // vLLM build switched from `reasoning_content` to `reasoning` for the
    // thinking-trace deltas at some point. Accept either via alias.
    #[serde(default, alias = "reasoning")] pub reasoning_content: Option<String>,
    #[serde(default)] pub tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIStreamToolCall {
    pub index: u32,
    #[serde(default)] pub id: Option<String>,
    #[serde(default, rename = "type")] pub kind: Option<String>,
    #[serde(default)] pub function: Option<OpenAIStreamToolCallFn>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct OpenAIStreamToolCallFn {
    #[serde(default)] pub name: Option<String>,
    #[serde(default)] pub arguments: Option<String>,
}

// ----- /v1/models OpenAI shape

#[derive(Debug, Serialize)]
pub struct OpenAIModelsResponse {
    pub object: &'static str,
    pub data: Vec<OpenAIModelEntry>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIModelEntry {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

// --------------------------------------------------------------- Anthropic ----

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<Value>, // string or [content blocks]
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicMessage {
    pub role: String, // "user" | "assistant"
    pub content: Value, // string OR Vec<AnthropicContentBlock>
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
    Image {
        source: Value,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Value, // string or [blocks]
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str, // "message"
    pub role: &'static str, // "assistant"
    pub model: String,
    pub content: Vec<AnthropicResponseBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicResponseBlock {
    Text { text: String },
    Thinking { thinking: String },
    ToolUse { id: String, name: String, input: Value },
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ----- /v1/models Anthropic shape

#[derive(Debug, Serialize)]
pub struct AnthropicModelsResponse {
    pub data: Vec<AnthropicModelEntry>,
    pub has_more: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AnthropicModelEntry {
    #[serde(rename = "type")]
    pub kind: &'static str, // "model"
    pub id: String,
    pub display_name: String,
    pub created_at: String, // RFC3339
}
