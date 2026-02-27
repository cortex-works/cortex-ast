//! Raw-text engine — handles **unstructured** plain-text files: logs, env-vars,
//! INI configs, and similar line-oriented content.
//!
//! IMPORTANT: `.yaml`, `.yml`, `.toml`, `.json`, `.md` are intentionally NOT
//! here.  Those structured formats are routed to `TreeSitterEngine` so that
//! the `cortex_act` patchers can reach their exact AST byte-offsets.
//!
//! `.sql` is routed here temporarily until a dedicated SQL-AST grammar is
//! integrated (TODO: upgrade to tree-sitter-sql Wasm driver).

use super::FileExplorer;
use anyhow::{Context, Result};
use std::path::Path;

const PREVIEW_LINES: usize = 30;
const TAIL_LINES: usize = 10;

/// Plain-text reader for log files, env files, text documents, etc.
pub struct RawTextEngine;

impl RawTextEngine {
    pub fn new() -> Self { Self }
}

impl FileExplorer for RawTextEngine {
    fn name(&self) -> &'static str { "raw_text" }

    fn supported_extensions(&self) -> &'static [&'static str] {
        // Strictly unstructured / line-oriented formats.
        // Structured formats (json/yaml/toml/md) go to TreeSitterEngine.
        // sql goes here until a tree-sitter-sql Wasm grammar is available.
        &["log", "txt", "env", "ini", "cfg", "conf", "sql"]
    }

    /// Preview: line count + first N lines + last M lines.
    fn get_overview(&self, path: &Path, max_rows: usize) -> Result<String> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;

        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();
        let head_n = max_rows.min(PREVIEW_LINES);

        let mut out = String::new();
        out.push_str(&format!(
            "# Text overview: {} ({} lines)\n\n",
            path.file_name().unwrap_or_default().to_string_lossy(),
            total,
        ));

        out.push_str(&format!("## First {} lines\n", head_n));
        for (i, line) in lines.iter().take(head_n).enumerate() {
            out.push_str(&format!("{:>5}  {}\n", i + 1, line));
        }

        if total > head_n + TAIL_LINES {
            out.push_str(&format!("\n… ({} lines omitted) …\n\n", total - head_n - TAIL_LINES));
            out.push_str(&format!("## Last {} lines\n", TAIL_LINES));
            for (i, line) in lines.iter().rev().take(TAIL_LINES).collect::<Vec<_>>().iter().rev().enumerate() {
                out.push_str(&format!("{:>5}  {}\n", total - TAIL_LINES + i + 1, line));
            }
        }

        Ok(out)
    }

    /// Read lines matching the query substring.  Returns all lines if query is
    /// empty or None.  Output is capped at `max_chars`.
    fn read_target(&self, path: &Path, query: Option<&str>, max_chars: usize) -> Result<String> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;

        let filter = query.unwrap_or("").to_lowercase();
        let mut out = String::new();

        for (i, line) in content.lines().enumerate() {
            if !filter.is_empty() && !line.to_lowercase().contains(&filter) {
                continue;
            }
            out.push_str(&format!("{:>5}  {}\n", i + 1, line));
            if out.len() >= max_chars {
                out.push_str(&format!("\n[output truncated at {} chars]\n", max_chars));
                break;
            }
        }

        if out.is_empty() {
            out.push_str("(no matching lines)\n");
        }

        Ok(out)
    }
}
