//! # Grammar Manager — Dynamic Wasm Language Plugin System
//!
//! Manages the lifecycle of tree-sitter grammar plugins for non-Core languages.
//! Core 3 (Rust, TypeScript, Python) are statically linked into the binary.
//! All other languages are served as `.wasm` grammars fetched from the CDN.
//!
//! ## Cache directory
//! `~/.cortex-works/grammars/<lang>.wasm`
//! `~/.cortex-works/grammars/<lang>_prune.scm`
//!
//! ## CDN base URL
//! `https://cdn.cortex-works.com/grammars/`

use anyhow::{Context, Result};
use std::path::PathBuf;

/// The three statically-compiled language names. They never need downloading.
pub const CORE_LANGUAGES: &[&str] = &["rust", "typescript", "python"];


// ─────────────────────────────────────────────────────────────────────────────
// Cache directory helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `~/.cortex-works/grammars/`, creating it if necessary.
pub fn grammar_cache_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("cannot resolve $HOME")?;
    let dir = home.join(".cortex-works").join("grammars");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating grammar cache dir: {}", dir.display()))?;
    Ok(dir)
}

/// Absolute path to the cached `.wasm` file for a language.
pub fn wasm_path(lang: &str) -> Result<PathBuf> {
    Ok(grammar_cache_dir()?.join(format!("{lang}.wasm")))
}

/// Absolute path to the cached `.scm` prune-query file for a language.
pub fn scm_path(lang: &str) -> Result<PathBuf> {
    Ok(grammar_cache_dir()?.join(format!("{lang}_prune.scm")))
}

// ─────────────────────────────────────────────────────────────────────────────
// Ensure grammar is available — the core function
// ─────────────────────────────────────────────────────────────────────────────

/// Ensure `{lang}.wasm` and `{lang}_prune.scm` exist in the local cache.
///
/// - If `lang` is one of [`CORE_LANGUAGES`] this is a no-op.
/// - Otherwise it checks the cache.  Missing files are downloaded from the CDN.
/// - Network or I/O errors are returned as `Err(...)`.  Callers should fall back
///   gracefully to the universal regex parser.
pub fn ensure_grammar_available(lang: &str) -> Result<()> {
    // Core languages are statically linked — nothing to do.
    if CORE_LANGUAGES.contains(&lang) {
        return Ok(());
    }

    let wasm = wasm_path(lang)?;
    let scm  = scm_path(lang)?;

    if !wasm.exists() {
        download_artifact(lang, "wasm", &wasm)?;
    }

    // The `.scm` file is optional: some grammars don't have body-pruning queries.
    // We attempt a download but don't fail if it returns 404.
    if !scm.exists() {
        let _ = download_artifact(lang, "scm", &scm);
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Download routine
// ─────────────────────────────────────────────────────────────────────────────

/// Map a language name to its GitHub release download URL.
/// Falls back to a predictable naming convention.
fn github_wasm_url(lang: &str) -> String {
    // Some langs have non-standard repo names or non-standard release asset names.
    match lang {
        "c_sharp"  => return "https://github.com/tree-sitter/tree-sitter-c-sharp/releases/latest/download/tree-sitter-c_sharp.wasm".to_string(),
        "cpp"      => return "https://github.com/tree-sitter/tree-sitter-cpp/releases/latest/download/tree-sitter-cpp.wasm".to_string(),
        "c"        => return "https://github.com/tree-sitter/tree-sitter-c/releases/latest/download/tree-sitter-c.wasm".to_string(),
        // yaml grammar is maintained by ikatyang, not the main tree-sitter org.
        "yaml"     => return "https://github.com/ikatyang/tree-sitter-yaml/releases/latest/download/tree-sitter-yaml.wasm".to_string(),
        // toml: nickel-lang maintains a wasm-releasing fork.
        "toml"     => return "https://github.com/nickel-lang/tree-sitter-toml/releases/latest/download/tree-sitter-toml.wasm".to_string(),
        // markdown: official tree-sitter org, asset uses hyphen not underscore.
        "markdown" => return "https://github.com/tree-sitter/tree-sitter-markdown/releases/latest/download/tree-sitter-markdown.wasm".to_string(),
        _ => {}
    }
    let repo_name = Box::leak(format!("tree-sitter-{lang}").into_boxed_str()) as &str;
    format!(
        "https://github.com/tree-sitter/{repo_name}/releases/latest/download/{repo_name}.wasm"
    )
}

/// Download a single grammar artifact and write it to `dest`.
/// Downloads from GitHub tree-sitter releases (primary source).
fn download_artifact(lang: &str, _kind: &str, dest: &PathBuf) -> Result<()> {
    // Only .wasm is downloaded; .scm files are optional and served from the same place
    let url = github_wasm_url(lang);

    eprintln!("[grammar_manager] Downloading {url} → {}", dest.display());

    let response = ureq::get(&url)
        .call()
        .with_context(|| format!("HTTP GET {url}"))?;

    let status = response.status();
    if status != 200 {
        anyhow::bail!("HTTP {status} fetching {url}");
    }

    let mut body: Vec<u8> = Vec::new();
    use std::io::Read;
    response
        .into_reader()
        .read_to_end(&mut body)
        .with_context(|| format!("reading response body from {url}"))?;

    std::fs::write(dest, &body)
        .with_context(|| format!("writing {}", dest.display()))?;

    eprintln!(
        "[grammar_manager] Saved {lang}.wasm ({} bytes) → {}",
        body.len(),
        dest.display()
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Query .scm content loader
// ─────────────────────────────────────────────────────────────────────────────

/// Read the body-prune query for a language from the local cache.
/// Returns `None` if no `.scm` file exists (grammar has no pruning queries).
pub fn load_prune_scm(lang: &str) -> Option<String> {
    let path = scm_path(lang).ok()?;
    std::fs::read_to_string(path).ok()
}

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap — seed embedded prune queries into the local cache
// ─────────────────────────────────────────────────────────────────────────────

/// Copy the prune `.scm` files that are embedded in the binary (via
/// `include_str!`) into the grammar cache directory so that Wasm drivers can
/// find them at runtime.
///
/// Each entry is `(embedded_content, cache_lang_name)`.  A file is only
/// written when it doesn't already exist, so user-supplied overrides are
/// preserved.
///
/// Call this once before `load_cached_wasm_drivers()` to ensure prune queries
/// are available on the very first run (before any `.wasm` download).
pub fn bootstrap_embedded_queries() {
    const QUERIES: &[(&str, &str)] = &[
        (include_str!("../queries/go_prune.scm"),     "go"),
        (include_str!("../queries/php_prune.scm"),    "php"),
        (include_str!("../queries/java_prune.scm"),   "java"),
        (include_str!("../queries/dart_prune.scm"),   "dart"),
        // Note: the repo file is named `csharp_prune.scm`; the grammar lang is `c_sharp`.
        (include_str!("../queries/csharp_prune.scm"), "c_sharp"),
        (include_str!("../queries/cpp_prune.scm"),    "cpp"),
        (include_str!("../queries/ruby_prune.scm"),   "ruby"),
        (include_str!("../queries/c_prune.scm"),      "c"),
    ];

    let Ok(dir) = grammar_cache_dir() else { return };

    for (content, lang) in QUERIES {
        let dest = dir.join(format!("{lang}_prune.scm"));
        if !dest.exists() {
            if let Err(e) = std::fs::write(&dest, content) {
                eprintln!("[grammar_manager] bootstrap: failed to write {}: {e}", dest.display());
            }
        }
    }
}
