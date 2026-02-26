use std::path::Path;
use anyhow::{Context, Result};
use crate::inspector::{exported_language_config, extract_symbols_from_source};
use crate::act::auto_healer::try_auto_heal;

pub struct AstEdit {
    /// e.g. "class:Auth" or "function:login" or just the bare identifier "login"
    pub target: String,
    pub action: String, // "replace_body", "replace", "delete"
    pub code: String,
}

pub fn apply_ast_edits(
    file_path: &Path,
    edits: Vec<AstEdit>,
    llm_url: Option<&str>,
) -> Result<String> {
    // 0. Permission Guard — fail fast before touching anything
    check_write_permission(file_path)?;

    let source_bytes = std::fs::read(file_path).context("Failed to read original source")?;
    let mut current_source = String::from_utf8_lossy(&source_bytes).into_owned();

    // 1. Initial Pass: gather targeted byte ranges Bottom-Up so string slicing applies safely
    let mut operations = Vec::new();
    let symbols = extract_symbols_from_source(file_path, &current_source);

    for edit in edits {
        // Simple search: target can be exactly "kind:name" or just "name"
        let sym = symbols.iter().find(|s| {
            let full_name = format!("{}:{}", s.kind, s.name);
            edit.target == full_name || edit.target == s.name
        });

        if let Some(s) = sym {
            let start = s.start_byte;
            let end = s.end_byte;
            operations.push((start, end, edit));
        } else {
            anyhow::bail!("AST Target not found in source: {}", edit.target);
        }
    }

    // Sort by start_byte in Descending Order (Bottom-Up)
    operations.sort_by(|a, b| b.0.cmp(&a.0));

    // 2. Apply Edits In-Memory
    for (start, end, edit) in operations {
        let prefix = &current_source[..start];
        let suffix = &current_source[end..];

        let replacement = match edit.action.as_str() {
            "replace" => edit.code.as_str(),
            "delete" => "",
            // Provide a fast "replace_body" fallback, but standard "replace" covers the node
            _ => edit.code.as_str(),
        };

        current_source = format!("{}{}{}", prefix, replacement, suffix);
    }

    // 3. Tree-sitter Validation / Virtual Dry-Run
    let cfg = exported_language_config().read().unwrap();
    let drv = cfg
        .driver_for_path(file_path)
        .with_context(|| format!("Tree-sitter driver not found for {:?}", file_path))?;

    let mut parser = drv.make_parser(file_path).context("Failed to load Wasm Parser")?;
    
    let parsed_tree = parser.parse(&current_source, None).context("Failed to parse altered code")?;
    
    // Check for ERROR nodes — collect human-readable error messages for Auto-Healer context
    if parsed_tree.root_node().has_error() {
        let ts_errors = collect_ts_errors(parsed_tree.root_node(), &current_source);
        eprintln!("[CortexAct] WARNING: AST Validation failed ({}). Invoking Auto-Healer...", ts_errors.join("; "));

        current_source = try_auto_heal(file_path, &current_source, &ts_errors, llm_url)?;

        // Final Double Check
        let final_tree = parser.parse(&current_source, None).context("Failed to parse healed code")?;
        if final_tree.root_node().has_error() {
            anyhow::bail!("Auto-Healer repaired code but it still contained Syntax Error nodes. Editor aborted safely.");
        }
    }

    // 4. Commit to Disk
    std::fs::write(file_path, &current_source).context("Failed to write to file")?;

    Ok(current_source)
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Verify the file is writable by the current process before any edits.
/// Returns a clear, actionable error if permissions are denied.
fn check_write_permission(path: &Path) -> Result<()> {
    let meta = std::fs::metadata(path)
        .with_context(|| format!("Cannot stat {:?} — file may not exist", path))?;

    if meta.permissions().readonly() {
        anyhow::bail!(
            "Permission denied: {:?} is read-only. \
             Run `chmod u+w {:?}` or check file ownership (expected user: zelda).",
            path, path
        );
    }

    // Attempt a zero-byte open for writing as the real test (catches ACL denials on macOS/Linux)
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .with_context(|| format!(
            "Write permission denied on {:?}. \
             Check file ownership and ACLs (ls -le {:?}).",
            path, path
        ))?;

    Ok(())
}

/// Walk the tree-sitter AST and collect human-readable descriptions of ERROR nodes.
/// These are passed to the Auto-Healer so the LLM knows *what* went wrong.
fn collect_ts_errors(node: tree_sitter::Node, source: &str) -> Vec<String> {
    let mut errors = Vec::new();
    collect_ts_errors_inner(node, source, &mut errors);
    errors
}

fn collect_ts_errors_inner(node: tree_sitter::Node, source: &str, out: &mut Vec<String>) {
    if node.is_error() || node.is_missing() {
        let row = node.start_position().row + 1;
        let col = node.start_position().column + 1;
        let snippet: String = source
            .get(node.start_byte()..node.end_byte())
            .unwrap_or("<unknown>")
            .chars()
            .take(40)
            .collect();
        if node.is_missing() {
            out.push(format!("Missing '{}' at line {}:{}", node.kind(), row, col));
        } else {
            out.push(format!("Unexpected token '{}' at line {}:{}", snippet.trim(), row, col));
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_ts_errors_inner(child, source, out);
    }
}

// ─── Unit Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ── Helper: create a temp file with given content ──────────────────────────
    fn temp_file(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new()
            .suffix(".rs")
            .tempfile()
            .expect("create temp file");
        f.write_all(content.as_bytes()).expect("write temp");
        f
    }

    // ── 1. Bottom-Up Sort ──────────────────────────────────────────────────────
    /// Prove that sorting by start_byte descending preserves later-byte offsets
    /// when multiple non-overlapping replacements are applied sequentially.
    #[test]
    fn bottom_up_sort_preserves_byte_offsets() {
        // Source: 3 clearly-delimited tokens at known byte positions
        let source = "AAAA BBBB CCCC";
        //            0    5    10

        // Simulate 3 replace operations with bytes known from source
        let mut ops: Vec<(usize, usize, &str)> = vec![
            (0, 4, "X"),    // replaces "AAAA" → "X"
            (5, 9, "Y"),    // replaces "BBBB" → "Y"
            (10, 14, "Z"),  // replaces "CCCC" → "Z"
        ];

        // Sort descending by start_byte (the Bottom-Up mandate)
        ops.sort_by(|a, b| b.0.cmp(&a.0));

        // Apply each replacement to a mutable String
        let mut buf = source.to_string();
        for (start, end, replacement) in ops {
            buf = format!("{}{}{}", &buf[..start], replacement, &buf[end..]);
        }

        assert_eq!(buf, "X Y Z",
            "Bottom-up replacement must produce 'X Y Z'; byte offsets corrupted if order wrong"
        );
    }

    /// Non-overlapping edits applied top-down WITHOUT sorting should corrupt offsets.
    /// This is the *failure mode* that bottom-up sorting prevents.
    #[test]
    fn top_down_order_corrupts_offsets() {
        let source = "AAAA BBBB CCCC";
        let ops: Vec<(usize, usize, &str)> = vec![
            (0, 4, "XX"),   // inserts extra char → shifts all following bytes
            (5, 9, "Y"),
            (10, 14, "Z"),
        ];
        // Apply top-down intentionally (no sort)
        let mut buf = source.to_string();
        for (start, end, replacement) in ops {
            // The second+ ops will panic or produce garbage if bytes shifted
            if end <= buf.len() {
                buf = format!("{}{}{}", &buf[..start], replacement, &buf[end..]);
            }
        }
        // Result should NOT equal what bottom-up correctly produces
        assert_ne!(buf, "XX Y Z",
            "Top-down is expected to corrupt. If this passes, the source offsets happened to align — add more skew."
        );
    }

    // ── 2. sanitize_llm_code (re-exported from auto_healer tests, proven here) ──
    // (see auto_healer.rs tests for the sanitizer itself)

    // ── 3. collect_ts_errors ───────────────────────────────────────────────────
    /// A deliberately broken Rust snippet should yield at least 1 TS error message
    /// with the expected line/col information.
    #[test]
    fn ts_error_collection_on_broken_rust() {
        use tree_sitter::Parser;

        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::language().into())
            .expect("load rust grammar");

        // Missing closing brace — classic syntax error
        let broken = "fn broken() { let x = 5;";
        let tree = parser.parse(broken, None).unwrap();

        assert!(tree.root_node().has_error(),
            "Parser must detect an error in the broken snippet");

        let errors = collect_ts_errors(tree.root_node(), broken);
        assert!(!errors.is_empty(),
            "collect_ts_errors must return at least one description");

        // Every error message must mention 'line' to be human-readable
        for e in &errors {
            assert!(e.contains("line"),
                "Error description should contain 'line': {:?}", e);
        }
    }

    // ── 4. check_write_permission ──────────────────────────────────────────────
    /// Passing a read-only file must produce a clear, actionable error.
    #[test]
    fn permission_guard_catches_readonly() {
        use std::os::unix::fs::PermissionsExt;

        let f = temp_file("fn main() {}");
        let path = f.path();

        // Make read-only
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o444))
            .expect("set readonly");

        let result = check_write_permission(path);
        assert!(result.is_err(), "Should fail for read-only file");
        let msg = format!("{:?}", result.unwrap_err());
        assert!(
            msg.contains("read-only") || msg.contains("Permission denied"),
            "Error must mention read-only or permission: {}", msg
        );

        // Restore to avoid tempfile cleanup panic
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o644)).ok();
    }

    /// A writable temp file must pass the permission guard.
    #[test]
    fn permission_guard_passes_for_writable() {
        let f = temp_file("fn main() {}");
        assert!(check_write_permission(f.path()).is_ok(),
            "Writable file should pass permission guard");
    }
}
