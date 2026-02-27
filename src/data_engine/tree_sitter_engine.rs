//! Tree-sitter-pipeline engine for structured markup and config files.
//!
//! These file types MUST go through the AST pipeline (tree-sitter) so that
//! `cortex_act` patchers can target exact byte-offsets of AST nodes.  Routing
//! them to `RawTextEngine` (flat-line treatment) would silently break surgical
//! patching via `cortex_act_edit_data_graph` and `cortex_act_edit_markup`.
//!
//! Handled extensions: `.json`, `.yaml`, `.yml`, `.toml`, `.md`, `.markdown`
//!
//! ## get_overview()
//! Produces a *structural* preview that mirrors what the tree-sitter AST sees:
//! - JSON   → top-level key/value type map (via `serde_json`)
//! - YAML   → top-level key/value type map (via `serde_yaml`)
//! - TOML   → section headers + key count (via `toml`)
//! - MD     → heading hierarchy extracted line-by-line
//!
//! ## read_target()
//! Returns the **raw file bytes as UTF-8** (up to `max_chars`), optionally
//! filtered by substring across lines.  Consumers that need byte-accurate AST
//! patching should work from the raw content returned here.

use super::FileExplorer;
use anyhow::{Context, Result};
use std::path::Path;

pub struct TreeSitterEngine;

impl TreeSitterEngine {
    pub fn new() -> Self { Self }

    fn ext(path: &Path) -> &str {
        path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
    }
}

impl FileExplorer for TreeSitterEngine {
    fn name(&self) -> &'static str { "tree_sitter" }

    fn supported_extensions(&self) -> &'static [&'static str] {
        &["json", "yaml", "yml", "toml", "md", "markdown"]
    }

    fn get_overview(&self, path: &Path, _max_rows: usize) -> Result<String> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let filename = path.file_name().unwrap_or_default().to_string_lossy();

        match Self::ext(path) {
            "json" => overview_json(&filename, &content),
            "yaml" | "yml" => overview_yaml(&filename, &content),
            "toml" => overview_toml(&filename, &content),
            "md" | "markdown" => overview_markdown(&filename, &content),
            _ => Ok(format!(
                "# {filename}\n\n(structural overview not available for this extension)\n"
            )),
        }
    }

    fn read_target(&self, path: &Path, query: Option<&str>, max_chars: usize) -> Result<String> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;

        let filter = query.unwrap_or("").to_lowercase();
        if filter.is_empty() {
            // Return full raw content — callers need byte-accurate text.
            if content.len() <= max_chars {
                return Ok(content);
            }
            let mut cut = max_chars;
            while cut > 0 && !content.is_char_boundary(cut) { cut -= 1; }
            let mut out = content[..cut].to_string();
            out.push_str(&format!("\n[truncated at {max_chars} chars]\n"));
            return Ok(out);
        }

        // Filtered: return matching lines with line numbers.
        let mut out = String::new();
        for (i, line) in content.lines().enumerate() {
            if line.to_lowercase().contains(&filter) {
                out.push_str(&format!("{:>5}  {}\n", i + 1, line));
                if out.len() >= max_chars {
                    out.push_str(&format!("\n[truncated at {max_chars} chars]\n"));
                    break;
                }
            }
        }
        if out.is_empty() {
            out.push_str("(no matching lines)\n");
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Structural overview helpers
// ─────────────────────────────────────────────────────────────────────────────

fn type_label(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null       => "null",
        serde_json::Value::Bool(_)    => "bool",
        serde_json::Value::Number(_)  => "number",
        serde_json::Value::String(_)  => "string",
        serde_json::Value::Array(a)   => if a.is_empty() { "[]" } else { "[…]" },
        serde_json::Value::Object(o)  => if o.is_empty() { "{}" } else { "{…}" },
    }
}

fn overview_json(filename: &str, content: &str) -> Result<String> {
    let mut out = format!("# JSON: {filename}\n\n");

    match serde_json::from_str::<serde_json::Value>(content) {
        Err(e) => {
            out.push_str(&format!("⚠ parse error: {e}\n\n"));
            out.push_str("(raw preview)\n");
            for (i, line) in content.lines().take(20).enumerate() {
                out.push_str(&format!("{:>4}  {line}\n", i + 1));
            }
        }
        Ok(serde_json::Value::Object(map)) => {
            out.push_str(&format!("Top-level object — {} keys\n\n", map.len()));
            for (k, v) in map.iter().take(60) {
                out.push_str(&format!("  {:.<40} {}\n", format!("{k} "), type_label(v)));
            }
            if map.len() > 60 {
                out.push_str(&format!("  … ({} more keys)\n", map.len() - 60));
            }
        }
        Ok(serde_json::Value::Array(arr)) => {
            out.push_str(&format!("Top-level array — {} items\n\n", arr.len()));
            if let Some(first) = arr.first() {
                out.push_str("First item:\n");
                out.push_str(&serde_json::to_string_pretty(first).unwrap_or_default());
                out.push('\n');
            }
        }
        Ok(other) => {
            out.push_str(&format!("Scalar: {other}\n"));
        }
    }
    Ok(out)
}

fn yaml_value_type(v: &serde_yaml::Value) -> &'static str {
    match v {
        serde_yaml::Value::Null       => "null",
        serde_yaml::Value::Bool(_)    => "bool",
        serde_yaml::Value::Number(_)  => "number",
        serde_yaml::Value::String(_)  => "string",
        serde_yaml::Value::Sequence(s) => if s.is_empty() { "[]" } else { "[…]" },
        serde_yaml::Value::Mapping(m)  => if m.is_empty() { "{}" } else { "{…}" },
        serde_yaml::Value::Tagged(_)   => "tagged",
    }
}

fn overview_yaml(filename: &str, content: &str) -> Result<String> {
    let mut out = format!("# YAML: {filename}\n\n");

    match serde_yaml::from_str::<serde_yaml::Value>(content) {
        Err(e) => {
            out.push_str(&format!("⚠ parse error: {e}\n\n(raw preview)\n"));
            for (i, line) in content.lines().take(20).enumerate() {
                out.push_str(&format!("{:>4}  {line}\n", i + 1));
            }
        }
        Ok(serde_yaml::Value::Mapping(map)) => {
            out.push_str(&format!("Top-level mapping — {} keys\n\n", map.len()));
            for (k, v) in map.iter().take(60) {
                let key = match k {
                    serde_yaml::Value::String(s) => s.clone(),
                    other => format!("{other:?}"),
                };
                out.push_str(&format!("  {:.<40} {}\n", format!("{key} "), yaml_value_type(v)));
            }
        }
        Ok(other) => {
            out.push_str(&format!("Document value: {other:?}\n"));
        }
    }
    Ok(out)
}

fn overview_toml(filename: &str, content: &str) -> Result<String> {
    let mut out = format!("# TOML: {filename}\n\n");

    match content.parse::<toml::Value>() {
        Err(e) => {
            out.push_str(&format!("⚠ parse error: {e}\n\n(raw preview)\n"));
            for (i, line) in content.lines().take(20).enumerate() {
                out.push_str(&format!("{:>4}  {line}\n", i + 1));
            }
        }
        Ok(toml::Value::Table(table)) => {
            out.push_str(&format!("Root table — {} keys\n\n", table.len()));
            for (k, v) in table.iter() {
                let (type_str, detail) = match v {
                    toml::Value::Table(t)   => ("table",   format!("({} keys)", t.len())),
                    toml::Value::Array(a)   => ("array",   format!("({} items)", a.len())),
                    toml::Value::String(s)  => ("string",  format!("= {:?}", s.chars().take(50).collect::<String>())),
                    toml::Value::Integer(n) => ("integer", format!("= {n}")),
                    toml::Value::Float(f)   => ("float",   format!("= {f}")),
                    toml::Value::Boolean(b) => ("bool",    format!("= {b}")),
                    toml::Value::Datetime(d)=> ("datetime",format!("= {d}")),
                };
                out.push_str(&format!("  [{k}] {type_str} {detail}\n"));
            }
        }
        Ok(other) => {
            out.push_str(&format!("Document: {other}\n"));
        }
    }
    Ok(out)
}

fn overview_markdown(filename: &str, content: &str) -> Result<String> {
    let mut out = format!("# Markdown: {filename}\n\n");

    let headings: Vec<(usize, &str)> = content
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim_start_matches('#');
            let level = line.len() - trimmed.len();
            if level >= 1 && level <= 6 && line.starts_with('#') {
                Some((level, trimmed.trim()))
            } else {
                None
            }
        })
        .collect();

    let total_lines = content.lines().count();
    let word_count: usize = content.split_whitespace().count();
    out.push_str(&format!("{total_lines} lines · ~{word_count} words · {} headings\n\n", headings.len()));

    if headings.is_empty() {
        out.push_str("(no headings found)\n");
    } else {
        out.push_str("## Heading outline\n\n");
        for (level, text) in &headings {
            let indent = "  ".repeat(level - 1);
            let marker = "#".repeat(*level);
            out.push_str(&format!("{indent}{marker} {text}\n"));
        }
    }
    Ok(out)
}
