use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Gauge, Paragraph, Row, Sparkline, Table, Wrap},
    Terminal,
};

use crate::metrics::{ChatState, SharedState};
use crate::recipe::Stack;

// ---- palette (cyan dgxtop default) ------------------------------------------
const ACCENT: Color = Color::Cyan;
const TEXT: Color = Color::Gray;
const VALUE: Color = Color::White;
const DIM: Color = Color::DarkGray;
const OK: Color = Color::Green;
const WARN: Color = Color::Yellow;
const BAD: Color = Color::Red;

// ---- ui state ---------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode { Overview, Details, Chat }

struct Ui {
    mode: Mode,
    selected: usize,
    /// When true, expand the thinking trace under the "thought for N.Ns" header.
    /// Toggle with Ctrl-O from any mode. Defaults to collapsed.
    show_thinking: bool,
}

pub async fn run(stack: Arc<Stack>, state: SharedState) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let res = event_loop(&mut terminal, stack.clone(), state.clone()).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}

async fn event_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    stack: Arc<Stack>,
    state: SharedState,
) -> Result<()> {
    let mut ui = Ui { mode: Mode::Overview, selected: 0, show_thinking: false };
    let tick = Duration::from_millis(250);
    let mut last_tick = Instant::now();
    let openai_base = format!("http://127.0.0.1:{}", stack.proxy.openai_port);

    loop {
        // Clamp selection in case the model list shrinks.
        let n = stack.models.len().max(1);
        if ui.selected >= n { ui.selected = n - 1; }

        terminal.draw(|f| draw(f, &stack, &state, &ui))?;
        let timeout = tick.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press { continue; }
                if matches!(ui.mode, Mode::Chat) {
                    if !handle_chat_key(&k, &mut ui, &stack, &state, &openai_base) {
                        return Ok(());
                    }
                } else {
                    if !handle_nav_key(&k, &mut ui, &stack) {
                        return Ok(());
                    }
                }
            }
        }
        if last_tick.elapsed() >= tick { last_tick = Instant::now(); }
    }
}

// Returns false to request quit.
fn handle_nav_key(k: &crossterm::event::KeyEvent, ui: &mut Ui, stack: &Stack) -> bool {
    let n = stack.models.len().max(1);
    match k.code {
        KeyCode::Char('q') => return false,
        KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => return false,
        KeyCode::Char('o') if k.modifiers.contains(KeyModifiers::CONTROL) => {
            ui.show_thinking = !ui.show_thinking;
        }
        KeyCode::Esc => {
            if ui.mode == Mode::Overview { return false; }
            ui.mode = Mode::Overview;
        }
        KeyCode::Up | KeyCode::Char('k') => {
            if ui.selected > 0 { ui.selected -= 1; }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if ui.selected + 1 < n { ui.selected += 1; }
        }
        KeyCode::Enter => {
            if ui.mode == Mode::Overview { ui.mode = Mode::Details; }
        }
        KeyCode::Tab => match ui.mode {
            Mode::Overview => ui.mode = Mode::Details,
            Mode::Details => ui.mode = Mode::Chat,
            Mode::Chat => ui.mode = Mode::Details,
        },
        _ => {}
    }
    true
}

fn handle_chat_key(
    k: &crossterm::event::KeyEvent,
    ui: &mut Ui,
    stack: &Stack,
    state: &SharedState,
    openai_base: &str,
) -> bool {
    // Chat input takes most keys; only a few are commands.
    match k.code {
        KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => return false,
        KeyCode::Char('o') if k.modifiers.contains(KeyModifiers::CONTROL) => {
            ui.show_thinking = !ui.show_thinking;
            return true;
        }
        KeyCode::Esc => { ui.mode = Mode::Overview; return true; }
        KeyCode::Tab => { ui.mode = Mode::Details; return true; }
        KeyCode::PageUp => {
            let model = stack.models.get(ui.selected).map(|m| m.name.clone());
            if let Some(name) = model {
                let mut s = state.write();
                let c = s.chats.entry(name).or_default();
                // Scroll value is "lines from the top", so going UP visually
                // means a SMALLER offset. Also break auto-follow.
                c.scroll = c.scroll.saturating_sub(5);
                c.auto_follow = false;
            }
        }
        KeyCode::PageDown => {
            let model = stack.models.get(ui.selected).map(|m| m.name.clone());
            if let Some(name) = model {
                let mut s = state.write();
                let c = s.chats.entry(name).or_default();
                c.scroll = c.scroll.saturating_add(5);
                // draw() will detect when scroll has caught up to the bottom
                // and re-enable auto_follow.
            }
        }
        KeyCode::End => {
            let model = stack.models.get(ui.selected).map(|m| m.name.clone());
            if let Some(name) = model {
                let mut s = state.write();
                let c = s.chats.entry(name).or_default();
                c.auto_follow = true;
            }
        }
        KeyCode::Home => {
            let model = stack.models.get(ui.selected).map(|m| m.name.clone());
            if let Some(name) = model {
                let mut s = state.write();
                let c = s.chats.entry(name).or_default();
                c.scroll = 0;
                c.auto_follow = false;
            }
        }
        KeyCode::Backspace => {
            let model = stack.models.get(ui.selected).map(|m| m.name.clone());
            if let Some(name) = model {
                let mut s = state.write();
                let c = s.chats.entry(name).or_default();
                c.input.pop();
            }
        }
        KeyCode::Enter => {
            let model = stack.models.get(ui.selected).map(|m| m.name.clone());
            if let Some(name) = model {
                let in_flight = state.read().chats.get(&name).map(|c| c.in_flight).unwrap_or(false);
                let has_input = state.read().chats.get(&name).map(|c| !c.input.trim().is_empty()).unwrap_or(false);
                if !in_flight && has_input {
                    // Snap back to the latest content when the user sends.
                    {
                        let mut s = state.write();
                        if let Some(c) = s.chats.get_mut(&name) { c.auto_follow = true; }
                    }
                    crate::chat::submit(state.clone(), name, openai_base.to_string());
                }
            }
        }
        KeyCode::Char(c) => {
            let model = stack.models.get(ui.selected).map(|m| m.name.clone());
            if let Some(name) = model {
                let mut s = state.write();
                let cs = s.chats.entry(name).or_default();
                cs.input.push(c);
            }
        }
        _ => {}
    }
    true
}

// ---- drawing ----------------------------------------------------------------

fn draw(f: &mut ratatui::Frame, stack: &Stack, state: &SharedState, ui: &Ui) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),   // header
            Constraint::Length(1),   // separator
            Constraint::Min(8),      // body
            Constraint::Length(1),   // footer
        ])
        .split(area);

    draw_header(f, chunks[0], stack, state, ui);
    draw_separator(f, chunks[1]);
    match ui.mode {
        Mode::Overview => draw_overview(f, chunks[2], stack, state, ui),
        Mode::Details => draw_details(f, chunks[2], stack, state, ui),
        Mode::Chat => draw_chat(f, chunks[2], stack, state, ui),
    }
    draw_footer(f, chunks[3], state, ui);
}

// ---- header / footer / separator -------------------------------------------

fn draw_separator(f: &mut ratatui::Frame, area: Rect) {
    let line = "─".repeat(area.width as usize);
    f.render_widget(Paragraph::new(Span::styled(line, Style::default().fg(DIM))), area);
}

fn draw_header(f: &mut ratatui::Frame, area: Rect, stack: &Stack, state: &SharedState, ui: &Ui) {
    let s = state.read();
    let (badge, color) = if s.proxy.running { ("●", OK) } else { ("●", BAD) };
    let uptime = s.proxy.uptime.map(human_duration).unwrap_or_else(|| "—".into());
    let mode_label = match ui.mode {
        Mode::Overview => "overview",
        Mode::Details => "details",
        Mode::Chat => "chat",
    };

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(20), Constraint::Length(28)])
        .split(area);

    let left = Line::from(vec![
        Span::styled(badge.to_string(), Style::default().fg(color).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
        Span::styled("stackctl", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled("  ▸  ", Style::default().fg(DIM)),
        Span::styled(stack.name.clone(), Style::default().fg(VALUE).add_modifier(Modifier::BOLD)),
        Span::styled("    ", Style::default()),
        Span::styled("OAI", Style::default().fg(DIM)), Span::raw(" "),
        Span::styled(format!(":{}", stack.proxy.openai_port), Style::default().fg(TEXT)),
        Span::styled("   ANT ", Style::default().fg(DIM)),
        Span::styled(format!(":{}", stack.proxy.anthropic_port), Style::default().fg(TEXT)),
        Span::styled("    up ", Style::default().fg(DIM)),
        Span::styled(uptime, Style::default().fg(TEXT)),
    ]);
    f.render_widget(Paragraph::new(left), cols[0]);

    let right = Line::from(vec![
        Span::styled("mode ", Style::default().fg(DIM)),
        Span::styled(mode_label, Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
    ]);
    f.render_widget(Paragraph::new(right).alignment(Alignment::Right), cols[1]);
}

fn draw_footer(f: &mut ratatui::Frame, area: Rect, state: &SharedState, ui: &Ui) {
    let s = state.read();
    let mut spans = vec![
        Span::styled("req", Style::default().fg(DIM)),
        Span::raw(" "),
        Span::styled(format!("{}", s.proxy.requests_total), Style::default().fg(VALUE).add_modifier(Modifier::BOLD)),
        Span::raw("   "),
    ];
    if !s.proxy.requests_by_status.is_empty() {
        let mut keys: Vec<&u16> = s.proxy.requests_by_status.keys().collect();
        keys.sort();
        for k in keys {
            let count = s.proxy.requests_by_status[k];
            let color = if (*k) >= 500 { BAD } else if (*k) >= 400 { WARN } else { OK };
            spans.push(Span::styled(format!("{}", k), Style::default().fg(color)));
            spans.push(Span::styled(format!(" {}  ", count), Style::default().fg(VALUE)));
        }
    }
    if let Some(t) = s.proxy.last_request_at {
        spans.push(Span::styled(format!("last {} ago", human_duration(t.elapsed())), Style::default().fg(DIM)));
    }

    let thinking_label = if ui.show_thinking { "[ctrl-o] hide thinking" } else { "[ctrl-o] show thinking" };
    let hint = match ui.mode {
        Mode::Overview => format!("[↑↓] select   [enter/tab] details   {thinking_label}   [q] quit"),
        Mode::Details => format!("[tab] chat   [esc] overview   [↑↓] select   {thinking_label}   [q] quit"),
        Mode::Chat => format!("[enter] send   [tab] details   [esc] overview   {thinking_label}   [pgup/pgdn] scroll"),
    };
    let hint_w = hint.chars().count() as u16 + 1;
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(20), Constraint::Length(hint_w)])
        .split(area);
    f.render_widget(Paragraph::new(Line::from(spans)), cols[0]);
    f.render_widget(
        Paragraph::new(Span::styled(hint, Style::default().fg(DIM))).alignment(Alignment::Right),
        cols[1],
    );
}

// ---- overview (host + models table) ----------------------------------------

fn draw_overview(f: &mut ratatui::Frame, area: Rect, stack: &Stack, state: &SharedState, ui: &Ui) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(2)])
        .split(area);
    draw_host(f, chunks[0], stack, state);
    draw_models_table(f, chunks[1], stack, state, ui.selected);
}

fn util_color(v: f64) -> Color {
    if v > 85.0 { BAD } else if v > 60.0 { WARN } else { OK }
}
fn pct_color(p: u16) -> Color {
    if p > 85 { BAD } else if p > 70 { WARN } else { OK }
}

fn draw_host(f: &mut ratatui::Frame, area: Rect, stack: &Stack, state: &SharedState) {
    let s = state.read();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);
    f.render_widget(section_title("HOST", &stack.host), rows[0]);

    let util = s.host.gpu_util.unwrap_or(0.0);
    let util_pct = util.clamp(0.0, 100.0) as u16;
    let util_value = if s.host.gpu_util.is_some() { format!("{:>5.1} %", util) } else { " n/a  ".into() };
    bar_row(f, rows[1], "GPU", util_pct, &util_value, util_color(util));

    let used = s.host.mem_used_gb.unwrap_or(0.0);
    let total = s.host.mem_total_gb.unwrap_or(0.0);
    let mem_pct = if total > 0.0 { ((used / total) * 100.0).clamp(0.0, 100.0) as u16 } else { 0 };
    let mem_value = if total > 0.0 { format!("{:>5.1} / {:>5.1} GB", used, total) } else { " n/a ".into() };
    bar_row(f, rows[2], "MEM", mem_pct, &mem_value, pct_color(mem_pct));
}

fn section_title<'a>(name: &'a str, sub: &'a str) -> Paragraph<'a> {
    Paragraph::new(Line::from(vec![
        Span::styled("▎", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
        Span::styled(name, Style::default().fg(VALUE).add_modifier(Modifier::BOLD)),
        Span::raw("   "),
        Span::styled(sub, Style::default().fg(DIM)),
    ]))
}

fn bar_row(f: &mut ratatui::Frame, area: Rect, label: &str, percent: u16, value: &str, color: Color) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(6),
            Constraint::Min(12),
            Constraint::Length(22),
        ])
        .split(area);
    f.render_widget(
        Paragraph::new(Span::styled(label.to_string(), Style::default().fg(DIM))),
        cols[0],
    );
    let bar = Gauge::default()
        .label("")
        .gauge_style(Style::default().fg(color))
        .percent(percent);
    f.render_widget(bar, cols[1]);
    f.render_widget(
        Paragraph::new(Span::styled(value.to_string(), Style::default().fg(VALUE)))
            .alignment(Alignment::Right),
        cols[2],
    );
}

fn draw_models_table(f: &mut ratatui::Frame, area: Rect, stack: &Stack, state: &SharedState, selected: usize) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1), Constraint::Min(2)])
        .split(area);
    f.render_widget(
        section_title("MODELS", &format!("{} on {}", stack.models.len(), stack.host)),
        chunks[0],
    );

    const W_ACTIVE: usize = 7;
    const W_WAIT: usize = 6;
    const W_PROMPT: usize = 11;
    const W_DECODE: usize = 11;
    const W_KV: usize = 7;
    const W_UPTIME: usize = 9;

    let header = Row::new(vec![
        Cell::from("  MODEL"),
        Cell::from("ST"),
        Cell::from("PORT"),
        Cell::from(format!("{:>w$}", "ACTIVE", w = W_ACTIVE)),
        Cell::from(format!("{:>w$}", "WAIT", w = W_WAIT)),
        Cell::from(format!("{:>w$}", "PROMPT t/s", w = W_PROMPT)),
        Cell::from(format!("{:>w$}", "DECODE t/s", w = W_DECODE)),
        Cell::from(format!("{:>w$}", "KV %", w = W_KV)),
        Cell::from(format!("{:>w$}", "UPTIME", w = W_UPTIME)),
    ])
    .style(Style::default().fg(DIM).add_modifier(Modifier::BOLD));

    let s = state.read();
    let rows: Vec<Row> = stack.models.iter().enumerate().map(|(i, m)| {
        let ms = s.models.get(&m.name).cloned().unwrap_or_default();
        let (badge, badge_color) = if ms.healthy { ("UP", OK) } else { ("DOWN", BAD) };
        let kv_pct = (ms.gpu_cache_usage_perc * 100.0).clamp(0.0, 100.0);
        let kv_color = pct_color(kv_pct.round() as u16);
        let uptime = ms.ready_since.map(|t| t.elapsed()).map(human_duration).unwrap_or_else(|| "—".into());
        let marker = if i == selected { "▶ " } else { "  " };
        let name_style = if i == selected {
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default().fg(VALUE).add_modifier(Modifier::BOLD)
        };
        Row::new(vec![
            Cell::from(format!("{}{}", marker, short_name(&m.name))).style(name_style),
            Cell::from(badge).style(Style::default().fg(badge_color).add_modifier(Modifier::BOLD)),
            Cell::from(format!("{}", m.port)).style(Style::default().fg(TEXT)),
            Cell::from(format!("{:>w$}", ms.num_requests_running as u64, w = W_ACTIVE)).style(Style::default().fg(VALUE)),
            Cell::from(format!("{:>w$}", ms.num_requests_waiting as u64, w = W_WAIT)).style(Style::default().fg(VALUE)),
            Cell::from(format!("{:>w$.1}", ms.prompt_rate, w = W_PROMPT)).style(Style::default().fg(VALUE)),
            Cell::from(format!("{:>w$.1}", ms.gen_rate, w = W_DECODE)).style(Style::default().fg(VALUE)),
            Cell::from(format!("{:>w$.1}", kv_pct, w = W_KV)).style(Style::default().fg(kv_color).add_modifier(Modifier::BOLD)),
            Cell::from(format!("{:>w$}", uptime, w = W_UPTIME)).style(Style::default().fg(TEXT)),
        ])
    }).collect();

    let widths = [
        Constraint::Min(22),
        Constraint::Length(4),
        Constraint::Length(5),
        Constraint::Length(W_ACTIVE as u16),
        Constraint::Length(W_WAIT as u16),
        Constraint::Length(W_PROMPT as u16),
        Constraint::Length(W_DECODE as u16),
        Constraint::Length(W_KV as u16),
        Constraint::Length(W_UPTIME as u16),
    ];
    let table = Table::new(rows, widths).header(header).column_spacing(2);
    f.render_widget(table, chunks[2]);
}

// ---- details view: per-model sparklines ------------------------------------

fn draw_details(f: &mut ratatui::Frame, area: Rect, stack: &Stack, state: &SharedState, ui: &Ui) {
    let model = match stack.models.get(ui.selected) {
        Some(m) => m,
        None => return,
    };
    let s = state.read();
    let ms = s.models.get(&model.name).cloned().unwrap_or_default();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // section title
            Constraint::Length(1), // status line
            Constraint::Length(1), // spacer
            Constraint::Min(6),    // sparklines (split below)
        ])
        .split(area);

    f.render_widget(section_title("MODEL", &model.name), chunks[0]);

    // status / metrics line
    let (badge, badge_color) = if ms.healthy { ("UP", OK) } else { ("DOWN", BAD) };
    let uptime = ms.ready_since.map(|t| t.elapsed()).map(human_duration).unwrap_or_else(|| "—".into());
    let status = Line::from(vec![
        Span::styled(badge, Style::default().fg(badge_color).add_modifier(Modifier::BOLD)),
        Span::styled(format!("  :{}", model.port), Style::default().fg(TEXT)),
        Span::styled("    active ", Style::default().fg(DIM)),
        Span::styled(format!("{}", ms.num_requests_running as u64), Style::default().fg(VALUE)),
        Span::styled("    wait ", Style::default().fg(DIM)),
        Span::styled(format!("{}", ms.num_requests_waiting as u64), Style::default().fg(VALUE)),
        Span::styled("    KV ", Style::default().fg(DIM)),
        Span::styled(format!("{:.1}%", ms.gpu_cache_usage_perc * 100.0), Style::default().fg(VALUE)),
        Span::styled("    uptime ", Style::default().fg(DIM)),
        Span::styled(uptime, Style::default().fg(VALUE)),
    ]);
    f.render_widget(Paragraph::new(status), chunks[1]);

    // Three sparkline panels, stacked.
    let sl = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
        ])
        .split(chunks[3]);

    let prompt: Vec<u64> = ms.history.iter().map(|h| (h.prompt_rate * 10.0).round() as u64).collect();
    let decode: Vec<u64> = ms.history.iter().map(|h| (h.gen_rate * 10.0).round() as u64).collect();
    let kv: Vec<u64> = ms.history.iter().map(|h| (h.kv_pct * 1000.0).round() as u64).collect();

    sparkline_panel(f, sl[0], "PROMPT tok/s", &prompt, &format!("{:.1}", ms.prompt_rate), OK);
    sparkline_panel(f, sl[1], "DECODE tok/s", &decode, &format!("{:.1}", ms.gen_rate), ACCENT);
    sparkline_panel(f, sl[2], "KV CACHE %", &kv, &format!("{:.1}", ms.gpu_cache_usage_perc * 100.0), pct_color((ms.gpu_cache_usage_perc * 100.0).round() as u16));
}

fn sparkline_panel(f: &mut ratatui::Frame, area: Rect, label: &str, data: &[u64], current: &str, color: Color) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM))
        .title(Line::from(vec![
            Span::raw(" "),
            Span::styled(label, Style::default().fg(DIM).add_modifier(Modifier::BOLD)),
            Span::styled(format!("   now "), Style::default().fg(DIM)),
            Span::styled(current.to_string(), Style::default().fg(VALUE).add_modifier(Modifier::BOLD)),
            Span::raw(" "),
        ]));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if data.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("(no data yet)", Style::default().fg(DIM))),
            inner,
        );
        return;
    }
    let sp = Sparkline::default()
        .data(data)
        .style(Style::default().fg(color))
        .bar_set(symbols::bar::NINE_LEVELS);
    f.render_widget(sp, inner);
}

// ---- chat view --------------------------------------------------------------

fn draw_chat(f: &mut ratatui::Frame, area: Rect, stack: &Stack, state: &SharedState, ui: &Ui) {
    let model = match stack.models.get(ui.selected) {
        Some(m) => m,
        None => return,
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // title
            Constraint::Length(1), // status
            Constraint::Min(5),    // messages
            Constraint::Length(3), // input box
        ])
        .split(area);

    f.render_widget(section_title("CHAT", &model.name), chunks[0]);

    let cs = state.read().chats.get(&model.name).cloned().unwrap_or_default();
    let status = if let Some(err) = &cs.error {
        Line::from(vec![
            Span::styled("error ", Style::default().fg(BAD).add_modifier(Modifier::BOLD)),
            Span::styled(err.clone(), Style::default().fg(BAD)),
        ])
    } else if cs.in_flight {
        Line::from(Span::styled("· streaming…", Style::default().fg(ACCENT)))
    } else if cs.messages.is_empty() {
        Line::from(Span::styled("type a message and press Enter", Style::default().fg(DIM)))
    } else {
        Line::from(Span::styled(
            format!("{} message{} · ready", cs.messages.len(), if cs.messages.len() == 1 { "" } else { "s" }),
            Style::default().fg(DIM),
        ))
    };
    f.render_widget(Paragraph::new(status), chunks[1]);

    draw_chat_messages(f, chunks[2], state, &model.name, ui.show_thinking);
    draw_chat_input(f, chunks[3], &cs);
}

fn draw_chat_messages(f: &mut ratatui::Frame, area: Rect, state: &SharedState, model: &str, show_thinking: bool) {
    let cs = state.read().chats.get(model).cloned().unwrap_or_default();

    let block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(DIM));
    let inner = block.inner(area);
    f.render_widget(block, area);

    // The role label prefix is fixed at 8 cells: 6-char left-padded role
    // ("you   " / "model ") + "│ ". Reserve that and word-wrap content to fit.
    let prefix_w = 8u16;
    let content_w = inner.width.saturating_sub(prefix_w).max(20) as usize;

    let mut lines: Vec<Line> = Vec::new();
    for (i, msg) in cs.messages.iter().enumerate() {
        if i > 0 { lines.push(Line::from("")); }
        lines.extend(message_lines(msg, content_w, show_thinking));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "(no messages yet)", Style::default().fg(DIM),
        )));
    }

    let total_lines = lines.len();
    let viewport_height = inner.height as usize;
    let max_offset = total_lines.saturating_sub(viewport_height) as u16;

    let actual_scroll = {
        let mut s = state.write();
        let c = s.chats.entry(model.to_string()).or_default();
        if c.auto_follow {
            c.scroll = max_offset;
            max_offset
        } else if c.scroll >= max_offset {
            c.scroll = max_offset;
            c.auto_follow = true;
            max_offset
        } else {
            c.scroll
        }
    };

    // No paragraph-level wrap — we already split lines to fit content_w with
    // the proper continuation prefix.
    let p = Paragraph::new(lines).scroll((actual_scroll, 0));
    f.render_widget(p, inner);
}

fn message_lines(msg: &crate::metrics::ChatMessage, content_w: usize, show_thinking: bool) -> Vec<Line<'static>> {
    let (label, color) = match msg.role.as_str() {
        "user" => ("you", ACCENT),
        "assistant" => ("model", OK),
        other => (Box::leak(other.to_string().into_boxed_str()) as &str, DIM),
    };

    let mut out: Vec<Line> = Vec::new();
    let mut role_emitted = false;

    let prefix = |continuation: bool| -> Vec<Span<'static>> {
        if continuation {
            vec![Span::raw("      "), Span::styled("│ ", Style::default().fg(DIM))]
        } else {
            vec![
                Span::styled(format!("{:<6}", label), Style::default().fg(color).add_modifier(Modifier::BOLD)),
                Span::styled("│ ", Style::default().fg(DIM)),
            ]
        }
    };

    // Push one rendered content line with the role prefix attached.
    let mut push_with_prefix = |role_seen: &mut bool, content: Vec<Span<'static>>, out: &mut Vec<Line<'static>>| {
        let mut spans = prefix(*role_seen);
        spans.extend(content);
        out.push(Line::from(spans));
        *role_seen = true;
    };

    // ---- thinking block (assistant only) ----
    if !msg.thinking.is_empty() {
        // Header: always shown. Body: gated on show_thinking.
        let (header_text, header_color) = if msg.thinking_complete {
            let dur = match (msg.thinking_started_at, msg.thinking_ended_at) {
                (Some(s), Some(e)) => format!(" {:.1}s", e.duration_since(s).as_secs_f64()),
                _ => String::new(),
            };
            let hint = if show_thinking { " ▼" } else { " ▸ (Ctrl-O)" };
            (format!("▸ thought for{dur}{hint}"), DIM)
        } else {
            (format!("{} thinking…", spinner_frame()), ACCENT)
        };
        push_with_prefix(
            &mut role_emitted,
            vec![Span::styled(
                header_text,
                Style::default().fg(header_color).add_modifier(Modifier::ITALIC | Modifier::BOLD),
            )],
            &mut out,
        );

        // Body: only render when expanded (Ctrl-O). The streaming spinner in
        // the header is enough to show activity while collapsed.
        if show_thinking {
            let thinking_body = msg.thinking.trim_start_matches(['\n', '\r']);
            let thinking_style = Style::default().fg(DIM).add_modifier(Modifier::ITALIC);
            let avail = content_w.saturating_sub(2).max(10);
            for ln in thinking_body.split('\n') {
                if ln.is_empty() {
                    let mut spans = prefix(true); spans.push(Span::raw("  "));
                    out.push(Line::from(spans));
                    continue;
                }
                for w in textwrap::wrap(ln, avail) {
                    let mut spans = prefix(true);
                    spans.push(Span::raw("  "));
                    spans.push(Span::styled(w.into_owned(), thinking_style));
                    out.push(Line::from(spans));
                }
            }
        }
    }

    // ---- content block ----
    let body = msg.content.trim_start_matches(['\n', '\r']);
    if !body.is_empty() {
        if role_emitted {
            out.push(Line::from(prefix(true))); // blank separator
        }
        let default_style = Style::default().fg(VALUE);
        for spans in render_text_block(body, content_w, default_style) {
            push_with_prefix(&mut role_emitted, spans, &mut out);
        }
    } else if !role_emitted {
        push_with_prefix(
            &mut role_emitted,
            vec![Span::styled(
                format!("{} streaming…", spinner_frame()),
                Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
            )],
            &mut out,
        );
    }

    out
}

/// Render a chunk of message text into width-bounded styled lines.
///
/// Supports:
///   - Headings: `# H1`, `## H2`, `### H3`+ (rendered cyan-bold; H1 underlined)
///   - Horizontal rules: `---`, `***`, `___` on their own line
///   - Markdown tables (with row dividers and proportional shrinking)
///   - Unordered list items (`-`, `*`, `+`)
///   - Inline `**bold**`, `*italic*` / `_italic_`, and `` `code` ``
/// Each output `Vec<Span>` is one rendered line of content, ready to be
/// concatenated after the role prefix in `message_lines`.
fn render_text_block(text: &str, width: usize, default_style: Style) -> Vec<Vec<Span<'static>>> {
    let mut out: Vec<Vec<Span<'static>>> = Vec::new();
    let raw_lines: Vec<&str> = text.split('\n').collect();
    let mut i = 0;
    let w = width.max(10);
    while i < raw_lines.len() {
        let line = raw_lines[i];

        // Horizontal rule
        if is_horizontal_rule(line) {
            let bar = "─".repeat(w);
            out.push(vec![Span::styled(bar, Style::default().fg(DIM))]);
            i += 1;
            continue;
        }

        // ATX heading
        if let Some((level, body)) = parse_heading(line) {
            let style = heading_style(level).patch(default_style);
            for chunk in textwrap::wrap(&body, w) {
                out.push(inline_spans(&chunk, style));
            }
            i += 1;
            continue;
        }

        // Table (header + separator + rows)
        if is_table_row(line)
            && i + 1 < raw_lines.len()
            && is_table_separator(raw_lines[i + 1])
        {
            let start = i;
            i += 2;
            while i < raw_lines.len() && is_table_row(raw_lines[i]) { i += 1; }
            let rows: Vec<Vec<String>> = raw_lines[start..i].iter().enumerate()
                .filter(|(idx, _)| *idx != 1)
                .map(|(_, l)| parse_table_row(l))
                .collect();
            for tl in format_table(&rows, w) {
                out.push(vec![Span::styled(tl, Style::default().fg(TEXT))]);
            }
            continue;
        }

        // Unordered list item
        if let Some(item) = strip_bullet(line) {
            let bullet_str = "• ";
            let avail = w.saturating_sub(bullet_str.chars().count()).max(8);
            let wrapped = textwrap::wrap(&item, avail);
            for (j, chunk) in wrapped.into_iter().enumerate() {
                let mut spans = Vec::<Span<'static>>::new();
                spans.push(Span::styled(
                    if j == 0 { bullet_str.to_string() } else { "  ".to_string() },
                    Style::default().fg(ACCENT),
                ));
                spans.extend(inline_spans(&chunk, default_style));
                out.push(spans);
            }
            i += 1;
            continue;
        }

        // Plain paragraph
        if line.is_empty() {
            out.push(Vec::new());
        } else {
            for chunk in textwrap::wrap(line, w) {
                out.push(inline_spans(&chunk, default_style));
            }
        }
        i += 1;
    }
    if out.is_empty() { out.push(Vec::new()); }
    out
}

// Markdown helpers -----------------------------------------------------------

fn is_horizontal_rule(line: &str) -> bool {
    let t = line.trim();
    if t.len() < 3 { return false; }
    let first = t.chars().next().unwrap();
    matches!(first, '-' | '*' | '_')
        && t.chars().all(|c| c == first || c.is_whitespace())
        && t.chars().filter(|c| *c == first).count() >= 3
}

fn parse_heading(line: &str) -> Option<(usize, String)> {
    let t = line.trim_start();
    let level = t.chars().take_while(|&c| c == '#').count();
    if level == 0 || level > 6 { return None; }
    let rest = &t[level..];
    if !rest.starts_with(' ') { return None; }
    Some((level, rest.trim_start().to_string()))
}

fn heading_style(level: usize) -> Style {
    match level {
        1 => Style::default().fg(ACCENT).add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        2 => Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        3 => Style::default().fg(VALUE).add_modifier(Modifier::BOLD),
        _ => Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
    }
}

fn strip_bullet(line: &str) -> Option<String> {
    let t = line.trim_start();
    for marker in ["- ", "* ", "+ "] {
        if let Some(rest) = t.strip_prefix(marker) {
            return Some(rest.to_string());
        }
    }
    None
}

/// Parse inline markdown (bold/italic/code) into styled spans.
/// `default_style` is applied to plain text; markers add modifiers on top.
/// Unmatched markers are emitted as plain text so bad markdown still renders.
fn inline_spans(text: &str, default_style: Style) -> Vec<Span<'static>> {
    let mut out: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // **bold**
        if c == '*' && chars.get(i + 1) == Some(&'*') {
            // find next `**`
            let mut j = i + 2;
            let mut found = None;
            while j + 1 < chars.len() {
                if chars[j] == '*' && chars[j + 1] == '*' { found = Some(j); break; }
                j += 1;
            }
            if let Some(end) = found {
                if !buf.is_empty() { out.push(Span::styled(std::mem::take(&mut buf), default_style)); }
                let bold_text: String = chars[i + 2..end].iter().collect();
                out.push(Span::styled(bold_text, default_style.add_modifier(Modifier::BOLD)));
                i = end + 2;
                continue;
            }
        }
        // *italic* or _italic_
        if (c == '*' || c == '_') && chars.get(i + 1) != Some(&c) {
            // find matching closing marker on the same char
            let mut j = i + 1;
            let mut found = None;
            while j < chars.len() {
                if chars[j] == c { found = Some(j); break; }
                j += 1;
            }
            if let Some(end) = found {
                if end > i + 1 {
                    if !buf.is_empty() { out.push(Span::styled(std::mem::take(&mut buf), default_style)); }
                    let italic_text: String = chars[i + 1..end].iter().collect();
                    out.push(Span::styled(italic_text, default_style.add_modifier(Modifier::ITALIC)));
                    i = end + 1;
                    continue;
                }
            }
        }
        // `code`
        if c == '`' {
            let mut j = i + 1;
            let mut found = None;
            while j < chars.len() {
                if chars[j] == '`' { found = Some(j); break; }
                j += 1;
            }
            if let Some(end) = found {
                if !buf.is_empty() { out.push(Span::styled(std::mem::take(&mut buf), default_style)); }
                let code_text: String = chars[i + 1..end].iter().collect();
                out.push(Span::styled(
                    code_text,
                    Style::default().fg(WARN).add_modifier(Modifier::BOLD),
                ));
                i = end + 1;
                continue;
            }
        }
        buf.push(c);
        i += 1;
    }
    if !buf.is_empty() { out.push(Span::styled(buf, default_style)); }
    if out.is_empty() { out.push(Span::raw("")); }
    out
}

fn is_table_row(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with('|') && t.matches('|').count() >= 2
}

/// Separator row of a markdown table: only -, :, |, and whitespace, with at
/// least one dash. Examples: `|---|---|`, `| :--- | ---: |`.
fn is_table_separator(line: &str) -> bool {
    let t = line.trim();
    if !t.starts_with('|') { return false; }
    let stripped = t.trim_matches('|');
    let cells: Vec<&str> = stripped.split('|').collect();
    if cells.is_empty() { return false; }
    cells.iter().all(|c| {
        let c = c.trim();
        !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':')
    })
}

fn parse_table_row(line: &str) -> Vec<String> {
    let t = line.trim();
    let inner = t.trim_start_matches('|').trim_end_matches('|');
    inner.split('|').map(|c| c.trim().to_string()).collect()
}

/// Render parsed cells as an aligned box. If the natural width exceeds the
/// available width, each column is truncated proportionally so the table
/// still fits within `width` cells.
fn format_table(rows: &[Vec<String>], width: usize) -> Vec<String> {
    if rows.is_empty() { return Vec::new(); }
    let ncols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if ncols == 0 { return Vec::new(); }

    // Initial column widths from natural content.
    let mut widths: Vec<usize> = (0..ncols).map(|c| {
        rows.iter().map(|r| r.get(c).map(|s| s.chars().count()).unwrap_or(0)).max().unwrap_or(0)
    }).collect();

    // Box drawing cost: "│ " before each cell + " " after last cell + final "│"
    //   = 2*ncols + 2  ? actually it's: "│ <cell> │ <cell> │" => 1 + ncols*(width+3) + (-1 for last sep)
    //   Simpler: outer "│ " + cells joined by " │ " + " │"
    let chrome_per_row = 4 + 3 * (ncols.saturating_sub(1)); // "│ " + " │" + (ncols-1) * " │ "
    let natural: usize = widths.iter().sum::<usize>() + chrome_per_row;

    if natural > width && width > chrome_per_row {
        let budget = width - chrome_per_row;
        let total: usize = widths.iter().sum();
        if total > 0 {
            for w in widths.iter_mut() {
                *w = ((*w as f64 / total as f64) * budget as f64).floor() as usize;
                if *w < 1 { *w = 1; }
            }
            // Distribute the remainder so the sum equals budget.
            let mut remainder = budget.saturating_sub(widths.iter().sum::<usize>());
            for w in widths.iter_mut() {
                if remainder == 0 { break; }
                *w += 1;
                remainder -= 1;
            }
        }
    }

    let mut out = Vec::new();
    let border = |left: char, mid: char, right: char| -> String {
        let mut s = String::new();
        s.push(left);
        for (i, w) in widths.iter().enumerate() {
            if i > 0 { s.push(mid); }
            for _ in 0..(*w + 2) { s.push('─'); }
        }
        s.push(right);
        s
    };

    out.push(border('┌', '┬', '┐'));
    for (i, row) in rows.iter().enumerate() {
        let mut s = String::from("│");
        for (j, w) in widths.iter().enumerate() {
            let cell = row.get(j).map(String::as_str).unwrap_or("");
            let truncated: String = if cell.chars().count() > *w {
                let head: String = cell.chars().take(w.saturating_sub(1)).collect();
                format!("{head}…")
            } else {
                cell.to_string()
            };
            let pad = w.saturating_sub(truncated.chars().count());
            s.push(' ');
            s.push_str(&truncated);
            for _ in 0..pad { s.push(' '); }
            s.push(' ');
            s.push('│');
        }
        out.push(s);
        // Divider between every pair of consecutive rows (header and data).
        if i + 1 < rows.len() {
            out.push(border('├', '┼', '┤'));
        }
    }
    out.push(border('└', '┴', '┘'));
    out
}

fn spinner_frame() -> char {
    const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    FRAMES[(ms / 100) as usize % FRAMES.len()]
}

fn draw_chat_input(f: &mut ratatui::Frame, area: Rect, cs: &ChatState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if cs.in_flight { DIM } else { ACCENT }))
        .title(Line::from(vec![
            Span::raw(" "),
            Span::styled("INPUT", Style::default().fg(if cs.in_flight { DIM } else { ACCENT }).add_modifier(Modifier::BOLD)),
            Span::raw(" "),
        ]));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let cursor = if cs.in_flight { "" } else { "▏" };
    let p = Paragraph::new(Line::from(vec![
        Span::styled(cs.input.clone(), Style::default().fg(VALUE)),
        Span::styled(cursor, Style::default().fg(ACCENT)),
    ])).wrap(Wrap { trim: false });
    f.render_widget(p, inner);
}

// ---- helpers ----------------------------------------------------------------

fn short_name(name: &str) -> String {
    name.split('/').last().unwrap_or(name).to_string()
}

fn human_duration(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 { return format!("{}s", s); }
    if s < 3600 { return format!("{}m{:02}s", s / 60, s % 60); }
    format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
}
