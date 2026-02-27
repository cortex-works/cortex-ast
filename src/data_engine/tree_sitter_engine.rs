//! TRUE tree-sitter engine for structured markup and config files.
//!
//! Uses the same `WasmDriver` / `language_config()` pipeline as every other
//! AST language.  Grammar must be downloaded via:
//!   cortex_manage_ast_languages(action:"add", languages:["json","yaml","toml","markdown"])
//!
//! ## Why NOT serde (the previous fake implementation)
//!
//! serde does: parse → mutate → reserialize. Reserialisation DESTROYS ALL COMMENTS.
//!
//! tree-sitter gives `(start_byte, end_byte)` for every AST node.
//! A comment-preserving surgical patch is three byte-slice concatenations:
//!   raw[..start_byte]  +  new_value_bytes  +  raw[end_byte..]
//! Every byte outside the target node (comments, blank lines, formatting) is
//! left completely untouched.
//!
//! `find_node_bytes(path, dot_path)` is the public entry-point for patchers.

use super::FileExplorer;
use anyhow::{anyhow, Context, Result};
use std::path::Path;
use tree_sitter::Node;

pub struct TreeSitterEngine;

impl TreeSitterEngine {
    pub fn new() -> Self { Self }

    fn ext(path: &Path) -> String {
        path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase()
    }

    fn require_driver(path: &Path) -> Result<()> {
        let cfg = crate::inspector::exported_language_config();
        let guard = cfg.read().unwrap();
        if guard.driver_for_path(path).is_none() {
            let ext = Self::ext(path);
            let lang = ext_to_lang(&ext).to_string();
            anyhow::bail!(
                "No tree-sitter grammar loaded for .{ext}. \
                 Run: cortex_manage_ast_languages(action:\"add\", languages:[\"{lang}\"])"
            );
        }
        Ok(())
    }

    /// Walk the tree-sitter AST for `path` following `dot_path` (e.g. `"db.host"`)
    /// and return `(start_byte, end_byte)` of the VALUE node.
    ///
    /// Surgical splice: `raw[..start]  +  replacement  +  raw[end..]`
    /// Preserves ALL surrounding content — comments, blank lines, formatting.
    pub fn find_node_bytes(path: &Path, dot_path: &str) -> Result<Option<(usize, usize)>> {
        let source = std::fs::read(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let cfg = crate::inspector::exported_language_config();
        let guard = cfg.read().unwrap();
        let driver = guard
            .driver_for_path(path)
            .ok_or_else(|| anyhow!("No grammar for {}", path.display()))?;
        let source_text = std::str::from_utf8(&source)?;
        let mut parser = driver.make_parser(path)?;
        let tree = parser
            .parse(source_text, None)
            .ok_or_else(|| anyhow!("tree-sitter parse failed"))?;
        let keys: Vec<&str> = dot_path.split('.').collect();
        Ok(match Self::ext(path).as_str() {
            "json" => json_find_value(tree.root_node(), &source, &keys),
            "yaml" | "yml" => yaml_find_value(tree.root_node(), &source, &keys),
            "toml" => toml_find_value(tree.root_node(), &source, &keys),
            "md" | "markdown" => md_find_section_body(tree.root_node(), &source, dot_path),
            _ => None,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FileExplorer impl
// ─────────────────────────────────────────────────────────────────────────────

impl FileExplorer for TreeSitterEngine {
    fn name(&self) -> &'static str {
        "tree_sitter"
    }

    fn supported_extensions(&self) -> &'static [&'static str] {
        &["json", "yaml", "yml", "toml", "md", "markdown"]
    }

    fn get_overview(&self, path: &Path, _max_rows: usize) -> Result<String> {
        Self::require_driver(path)?;
        let source = std::fs::read(path)?;
        let source_text = std::str::from_utf8(&source)?;
        let cfg = crate::inspector::exported_language_config();
        let guard = cfg.read().unwrap();
        let driver = guard.driver_for_path(path).unwrap();
        let mut parser = driver.make_parser(path)?;
        let tree = parser
            .parse(source_text, None)
            .ok_or_else(|| anyhow!("parse failed: {}", path.display()))?;
        let filename = path.file_name().unwrap_or_default().to_string_lossy();
        let root = tree.root_node();
        match Self::ext(path).as_str() {
            "json" => overview_json(&filename, root, &source),
            "yaml" | "yml" => overview_yaml(&filename, root, &source),
            "toml" => overview_toml(&filename, root, &source),
            "md" | "markdown" => overview_markdown(&filename, root, &source, source_text),
            ext => Ok(format!("# {filename}\n(no structural view for .{ext})\n")),
        }
    }

    fn read_target(&self, path: &Path, query: Option<&str>, max_chars: usize) -> Result<String> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let filter = query.unwrap_or("").to_lowercase();
        if filter.is_empty() {
            if content.len() <= max_chars {
                return Ok(content);
            }
            let mut cut = max_chars;
            while cut > 0 && !content.is_char_boundary(cut) {
                cut -= 1;
            }
            return Ok(format!(
                "{}\n[truncated at {max_chars} chars]\n",
                &content[..cut]
            ));
        }
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
// Structural overview builders (use real AST node byte ranges)
// ─────────────────────────────────────────────────────────────────────────────

fn overview_json(filename: &str, root: Node, source: &[u8]) -> Result<String> {
    let mut out = format!("# JSON (tree-sitter): {filename}\n\n");
    if let Some(obj) = first_child_of_kind(root, "object")
        .or_else(|| first_descendant_of_kind(root, "object"))
    {
        let pairs = collect_pairs(obj, source);
        out.push_str(&format!("Top-level object — {} keys\n\n", pairs.len()));
        for (k, v) in &pairs {
            out.push_str(&format!(
                "  {:.<44} {} [{}..{}]\n",
                format!("{k} "),
                value_type_label(*v),
                v.start_byte(),
                v.end_byte()
            ));
        }
    } else if let Some(arr) = first_descendant_of_kind(root, "array") {
        out.push_str(&format!(
            "Top-level array — {} items\n",
            arr.named_child_count()
        ));
    } else {
        out.push_str("(empty or non-object root)\n");
    }
    Ok(out)
}

fn overview_yaml(filename: &str, root: Node, source: &[u8]) -> Result<String> {
    let mut out = format!("# YAML (tree-sitter): {filename}\n\n");
    if let Some(bm) = first_descendant_of_kind(root, "block_mapping") {
        let pairs = collect_pairs(bm, source);
        out.push_str(&format!("Root mapping — {} keys\n\n", pairs.len()));
        for (k, v) in &pairs {
            out.push_str(&format!(
                "  {:.<44} {} [{}..{}]\n",
                format!("{k} "),
                value_type_label(*v),
                v.start_byte(),
                v.end_byte()
            ));
        }
    } else {
        out.push_str("(no root block_mapping — grammar download needed?)\n");
    }
    Ok(out)
}

fn overview_toml(filename: &str, root: Node, source: &[u8]) -> Result<String> {
    let mut out = format!("# TOML (tree-sitter): {filename}\n\n");
    for i in 0..root.child_count() {
        let child = match root.child(i as u32) {
            Some(c) => c,
            None => continue,
        };
        match child.kind() {
            "key_value" | "pair" => {
                let k = node_key_text(child, source);
                let v = child.child_by_field_name("value").unwrap_or(child);
                out.push_str(&format!(
                    "  {k:<30} {} [{}..{}]\n",
                    value_type_label(v),
                    v.start_byte(),
                    v.end_byte()
                ));
            }
            "table" => {
                let heading = node_text(child, source)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string();
                out.push_str(&format!("\n{heading}\n"));
                for j in 0..child.child_count() {
                    let pair = match child.child(j as u32) {
                        Some(c) => c,
                        None => continue,
                    };
                    if matches!(pair.kind(), "key_value" | "pair") {
                        let k = node_key_text(pair, source);
                        let v = pair.child_by_field_name("value").unwrap_or(pair);
                        out.push_str(&format!(
                            "  {k:<30} {} [{}..{}]\n",
                            value_type_label(v),
                            v.start_byte(),
                            v.end_byte()
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn overview_markdown(
    filename: &str,
    root: Node,
    source: &[u8],
    src: &str,
) -> Result<String> {
    let mut out = format!("# Markdown (tree-sitter): {filename}\n\n");
    let lines = src.lines().count();
    let words = src.split_whitespace().count();
    let mut headings: Vec<(usize, String, usize, usize)> = Vec::new();
    collect_headings(root, source, &mut headings);
    out.push_str(&format!(
        "{lines} lines · ~{words} words · {} headings\n\n",
        headings.len()
    ));
    if headings.is_empty() {
        out.push_str("(no ATX headings found — grammar download needed?)\n");
    } else {
        out.push_str("## Heading outline (with byte ranges for surgical patching)\n\n");
        for (level, text, start, end) in &headings {
            let indent = "  ".repeat(level - 1);
            let marker = "#".repeat(*level);
            out.push_str(&format!(
                "{indent}{marker} {text}  [{}..{}]\n",
                start, end
            ));
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Surgical path finders (PUBLIC — used by config_patcher.rs)
// ─────────────────────────────────────────────────────────────────────────────

pub fn json_find_value(root: Node, source: &[u8], keys: &[&str]) -> Option<(usize, usize)> {
    let obj = first_child_of_kind(root, "object")
        .or_else(|| first_descendant_of_kind(root, "object"))?;
    descend_pairs(obj, source, keys)
}

pub fn yaml_find_value(root: Node, source: &[u8], keys: &[&str]) -> Option<(usize, usize)> {
    let bm = first_descendant_of_kind(root, "block_mapping")?;
    descend_pairs(bm, source, keys)
}

pub fn toml_find_value(root: Node, source: &[u8], keys: &[&str]) -> Option<(usize, usize)> {
    let (first, rest) = keys.split_first()?;
    for i in 0..root.child_count() {
        let child = match root.child(i as u32) {
            Some(c) => c,
            None => continue,
        };
        if matches!(child.kind(), "key_value" | "pair") {
            if node_key_text(child, source) == *first {
                let v = child.child_by_field_name("value")?;
                if rest.is_empty() {
                    return Some((v.start_byte(), v.end_byte()));
                }
                return descend_pairs(v, source, rest);
            }
        }
        if child.kind() == "table" {
            let heading = node_text(child, source)
                .lines()
                .next()
                .unwrap_or("")
                .to_string();
            let section = heading
                .trim_matches(|c| matches!(c, '[' | ']' | ' ' | '\n'));
            if section == *first {
                if rest.is_empty() {
                    return Some((child.start_byte(), child.end_byte()));
                }
                return descend_pairs(child, source, rest);
            }
        }
    }
    None
}

pub fn md_find_section_body(
    root: Node,
    source: &[u8],
    section: &str,
) -> Option<(usize, usize)> {
    let mut headings: Vec<(usize, String, usize, usize)> = Vec::new();
    collect_headings(root, source, &mut headings);
    let target = section.trim().to_lowercase();
    for (i, (level, text, _hs, he)) in headings.iter().enumerate() {
        if text.to_lowercase() == target {
            let body_start = *he;
            let body_end = headings
                .iter()
                .skip(i + 1)
                .find(|(l, _, _, _)| l <= level)
                .map(|(_, _, s, _)| *s)
                .unwrap_or(source.len());
            return Some((body_start, body_end));
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared traversal utilities
// ─────────────────────────────────────────────────────────────────────────────

fn descend_pairs<'a>(
    container: Node<'a>,
    source: &[u8],
    keys: &[&str],
) -> Option<(usize, usize)> {
    let (first, rest) = keys.split_first()?;
    for (k, v_node) in collect_pairs(container, source) {
        if k == *first {
            if rest.is_empty() {
                return Some((v_node.start_byte(), v_node.end_byte()));
            }
            return descend_pairs(unwrap_wrappers(v_node), source, rest);
        }
    }
    None
}

fn collect_pairs<'a>(container: Node<'a>, source: &[u8]) -> Vec<(String, Node<'a>)> {
    let mut out = Vec::new();
    for i in 0..container.child_count() {
        let child = match container.child(i as u32) {
            Some(c) => c,
            None => continue,
        };
        if matches!(
            child.kind(),
            "pair" | "block_mapping_pair" | "flow_pair" | "key_value"
        ) {
            if let (Some(kn), Some(vn)) = (
                child.child_by_field_name("key"),
                child.child_by_field_name("value"),
            ) {
                let key = leaf_text(kn, source)
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string();
                out.push((key, vn));
            }
        }
    }
    out
}

fn unwrap_wrappers(node: Node) -> Node {
    if matches!(node.kind(), "block_node" | "flow_node") {
        for i in 0..node.child_count() {
            if let Some(c) = node.child(i as u32) {
                if c.is_named() {
                    return c;
                }
            }
        }
    }
    node
}

fn leaf_text(node: Node, source: &[u8]) -> String {
    if node.child_count() == 0 {
        return node_text(node, source);
    }
    for i in 0..node.child_count() {
        if let Some(c) = node.child(i as u32) {
            if c.is_named() {
                return leaf_text(c, source);
            }
        }
    }
    node_text(node, source)
}

fn node_text(node: Node, source: &[u8]) -> String {
    std::str::from_utf8(&source[node.start_byte()..node.end_byte()])
        .unwrap_or("")
        .to_string()
}

fn node_key_text(pair: Node, source: &[u8]) -> String {
    if let Some(k) = pair.child_by_field_name("key") {
        return leaf_text(k, source)
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
    }
    for i in 0..pair.child_count() {
        if let Some(c) = pair.child(i as u32) {
            if c.is_named()
                && matches!(c.kind(), "bare_key" | "quoted_key" | "dotted_key")
            {
                return leaf_text(c, source);
            }
        }
    }
    String::new()
}

fn first_child_of_kind<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    (0..node.child_count()).find_map(|i| node.child(i as u32).filter(|c| c.kind() == kind))
}

fn first_descendant_of_kind<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    if node.kind() == kind {
        return Some(node);
    }
    for i in 0..node.child_count() {
        if let Some(c) = node.child(i as u32) {
            if let Some(found) = first_descendant_of_kind(c, kind) {
                return Some(found);
            }
        }
    }
    None
}

fn value_type_label(node: Node) -> &'static str {
    match node.kind() {
        "object" | "block_mapping" | "flow_mapping" | "inline_table" => "{object}",
        "array" | "block_sequence" | "flow_sequence" => "[array]",
        "string"
        | "double_quote_scalar"
        | "single_quote_scalar"
        | "plain_scalar"
        | "string_scalar"
        | "quoted_scalar" => "string",
        "number" | "integer" | "float" => "number",
        "true" | "false" | "boolean" => "bool",
        "null" => "null",
        _ => "value",
    }
}

fn collect_headings(
    node: Node,
    source: &[u8],
    out: &mut Vec<(usize, String, usize, usize)>,
) {
    if node.kind() == "atx_heading" {
        let raw = node_text(node, source);
        let trimmed = raw.trim_start_matches('#');
        let level = raw.len() - trimmed.len();
        out.push((
            level,
            trimmed.trim().to_string(),
            node.start_byte(),
            node.end_byte(),
        ));
        return;
    }
    if node.kind() == "setext_heading" {
        let raw = node_text(node, source);
        let level = if raw.trim().ends_with('=') { 1 } else { 2 };
        out.push((
            level,
            raw.lines().next().unwrap_or("").trim().to_string(),
            node.start_byte(),
            node.end_byte(),
        ));
        return;
    }
    for i in 0..node.child_count() {
        if let Some(c) = node.child(i as u32) {
            collect_headings(c, source, out);
        }
    }
}

pub fn ext_to_lang(ext: &str) -> &'static str {
    match ext {
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "md" | "markdown" => "markdown",
        _ => "unknown",
    }
}
