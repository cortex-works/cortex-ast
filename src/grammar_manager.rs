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
    // Some langs have non-standard repo names
    let repo_name = match lang {
        "c_sharp" => "tree-sitter-c-sharp",
        "cpp"     => "tree-sitter-cpp",
        "c"       => "tree-sitter-c",
        other     => Box::leak(format!("tree-sitter-{other}").into_boxed_str()),
    };
    // Asset filename uses lang identifier (c_sharp → c_sharp.wasm)
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
