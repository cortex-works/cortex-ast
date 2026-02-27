//! # Universal Data Engine
//!
//! A lightweight, extensible registry of file-format parsers that sit alongside
//! the tree-sitter AST engine.  The registry is used by:
//!
//! - `cortex_data_explorer` — query / preview CSV/TSV, structured markup,
//!   plain-text, and SQL files
//! - `cortex_get_capabilities` — enumerate all supported extensions and their
//!   assigned engine
//!
//! ## Extension routing (registration order = priority; first match wins)
//!
//! | Engine             | Extensions                              |
//! |--------------------|-----------------------------------------|
//! | `TreeSitterEngine` | `json yaml yml toml md markdown`        |
//! | `CsvEngine`        | `csv tsv`                               |
//! | `RawTextEngine`    | `log txt env ini cfg conf sql`          |
//!
//! ## Adding a new engine
//! 1. Create a new sub-module that implements [`FileExplorer`].
//! 2. Register it in [`ParserRegistry::default()`] **before** any fallback
//!    engine that might claim the same extension.

pub mod duckdb_engine;
pub mod raw_text_engine;
pub mod tree_sitter_engine;

use std::path::Path;
use anyhow::Result;

// ─────────────────────────────────────────────────────────────────────────────
// Core trait
// ─────────────────────────────────────────────────────────────────────────────

/// A file-format parser/explorer.  Implement this trait to teach CortexAST
/// about a new data format (CSV, Parquet, …) or text format (log, env, …).
pub trait FileExplorer: Send + Sync {
    /// Human-readable engine name (e.g. `"csv"`, `"raw_text"`).
    fn name(&self) -> &'static str;

    /// File extensions this engine handles (lowercase, no leading dot).
    fn supported_extensions(&self) -> &'static [&'static str];

    /// Return `true` when this engine can handle `path`.
    fn handles_path(&self, path: &Path) -> bool {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        self.supported_extensions().contains(&ext.as_str())
    }

    /// Return a short overview of the file: schema + first few rows (tabular),
    /// or line-count + head preview (text).
    fn get_overview(&self, path: &Path, max_rows: usize) -> Result<String>;

    /// Read content with optional filtering.
    ///
    /// - For tabular engines `query` may be a SQL-like `WHERE col = '…'` clause.
    /// - For text engines `query` is a substring/regex filter applied line-by-line.
    /// - `max_chars` caps output length.
    fn read_target(
        &self,
        path: &Path,
        query: Option<&str>,
        max_chars: usize,
    ) -> Result<String>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Registry
// ─────────────────────────────────────────────────────────────────────────────

/// Routes file paths to the correct [`FileExplorer`] engine.
pub struct ParserRegistry {
    engines: Vec<Box<dyn FileExplorer>>,
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self {
            // Registration order = routing priority (first match wins).
            // TreeSitterEngine must come before RawTextEngine so that
            // json/yaml/toml/md are captured before the plain-text fallback.
            engines: vec![
                Box::new(tree_sitter_engine::TreeSitterEngine::new()),
                Box::new(duckdb_engine::CsvEngine::new()),
                Box::new(raw_text_engine::RawTextEngine::new()),
            ],
        }
    }
}

impl ParserRegistry {
    /// Return the first engine that claims to handle `path`, or `None`.
    pub fn engine_for(&self, path: &Path) -> Option<&dyn FileExplorer> {
        self.engines.iter().find_map(|e| {
            if e.handles_path(path) {
                Some(e.as_ref())
            } else {
                None
            }
        })
    }

    /// Iterate over all registered engines (used by `cortex_get_capabilities`).
    pub fn engines(&self) -> &[Box<dyn FileExplorer>] {
        &self.engines
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global static registry (lazy-init, mirrors LanguageConfig pattern)
// ─────────────────────────────────────────────────────────────────────────────

use std::sync::OnceLock;

pub fn registry() -> &'static ParserRegistry {
    static REG: OnceLock<ParserRegistry> = OnceLock::new();
    REG.get_or_init(ParserRegistry::default)
}
