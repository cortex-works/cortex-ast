//! Comment-preserving config patcher.
//!
//! Uses `TreeSitterEngine::find_node_bytes(path, dot_path)` to locate the
//! exact byte range of the VALUE node, then splices in the replacement:
//!
//!   raw[..start_byte]  +  new_value_bytes  +  raw[end_byte..]
//!
//! Every byte outside the target node — YAML `#` comments, TOML inline
//! comments, blank lines, custom formatting — is left completely untouched.
//!
//! ## Fallback
//! If the relevant tree-sitter Wasm grammar is not yet loaded, the patcher
//! falls back to serde with a warning.  Install grammars once via:
//!   cortex_manage_ast_languages(action:"add", languages:["json","yaml","toml"])

use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::data_engine::tree_sitter_engine::TreeSitterEngine;

/// Patch a single key in a JSON, YAML, or TOML config file using dot-path
/// notation.  Comment-preserving surgical byte-splice when grammars are
/// loaded; serde fallback otherwise.
pub fn patch_config(
    file: &str,
    action: &str,
    dot_path: &str,
    value: Option<&serde_json::Value>,
) -> Result<String> {
    let path = Path::new(file);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "json" => patch_json(file, path, action, dot_path, value),
        "yaml" | "yml" => patch_yaml(file, path, action, dot_path, value),
        "toml" => patch_toml(file, path, action, dot_path, value),
        other => bail!("Unsupported config file extension: .{}", other),
    }
}

// ─── JSON ────────────────────────────────────────────────────────────────────

fn patch_json(
    file: &str,
    path: &Path,
    action: &str,
    dot_path: &str,
    value: Option<&serde_json::Value>,
) -> Result<String> {
    // ── Surgical path ──────────────────────────────────────────────────
    if action == "set" {
        if let Ok(Some((start, end))) = TreeSitterEngine::find_node_bytes(path, dot_path) {
            let v = value.context("'value' required for 'set' action")?;
            let new_bytes = serde_json::to_string(v).context("Serializing new JSON value")?;
            let raw = std::fs::read(file).context("Reading JSON file")?;
            let patched = [&raw[..start], new_bytes.as_bytes(), &raw[end..]].concat();
            std::fs::write(file, &patched).context("Writing JSON file")?;
            return Ok(format!(
                "✅ Patched JSON '{}' at '{}' [bytes {}..{}] (comment-preserving)",
                file, dot_path, start, end
            ));
        }
    }

    // ── serde fallback (delete, or grammar not loaded) ─────────────────
    // NOTE: JSON has no comments so serde round-trip is safe here.
    let raw = std::fs::read_to_string(file).context("Reading JSON file")?;
    let mut root: serde_json::Value =
        serde_json::from_str(&raw).context("Parsing JSON")?;
    let keys: Vec<&str> = dot_path.split('.').collect();
    let (parents, last) = keys.split_at(keys.len() - 1);
    let mut cursor = &mut root;
    for key in parents {
        cursor = cursor
            .get_mut(*key)
            .with_context(|| format!("Key '{}' not found in JSON", key))?;
    }
    match action {
        "set" => {
            let v = value.context("'value' required for 'set' action")?;
            cursor[last[0]] = v.clone();
        }
        "delete" => {
            if let Some(obj) = cursor.as_object_mut() {
                obj.remove(last[0]);
            }
        }
        other => bail!("Unknown action: {}", other),
    }
    let out = serde_json::to_string_pretty(&root).context("Serializing JSON")?;
    std::fs::write(file, &out).context("Writing JSON file")?;
    Ok(format!("✅ Patched JSON '{}' at '{}'", file, dot_path))
}

// ─── YAML ────────────────────────────────────────────────────────────────────

fn patch_yaml(
    file: &str,
    path: &Path,
    action: &str,
    dot_path: &str,
    value: Option<&serde_json::Value>,
) -> Result<String> {
    // ── Surgical path ──────────────────────────────────────────────────
    if action == "set" {
        if let Ok(Some((start, end))) = TreeSitterEngine::find_node_bytes(path, dot_path) {
            let v = value.context("'value' required for 'set' action")?;
            let new_bytes = value_to_yaml_scalar(v);
            let raw = std::fs::read(file).context("Reading YAML file")?;
            let patched = [&raw[..start], new_bytes.as_bytes(), &raw[end..]].concat();
            std::fs::write(file, &patched).context("Writing YAML file")?;
            return Ok(format!(
                "✅ Patched YAML '{}' at '{}' [bytes {}..{}] (comment-preserving)",
                file, dot_path, start, end
            ));
        }
    }

    // ── serde fallback (delete, or grammar not loaded) ─────────────────
    // ⚠ YAML comments are DESTROYED on serde round-trip.
    let raw = std::fs::read_to_string(file).context("Reading YAML file")?;
    let mut root: serde_yaml::Value =
        serde_yaml::from_str(&raw).context("Parsing YAML")?;
    let keys: Vec<&str> = dot_path.split('.').collect();
    let (parents, last) = keys.split_at(keys.len() - 1);
    let mut cursor = &mut root;
    for key in parents {
        cursor = cursor
            .get_mut(serde_yaml::Value::String(key.to_string()))
            .with_context(|| format!("Key '{}' not found in YAML", key))?;
    }
    match action {
        "set" => {
            let v = value.context("'value' required for 'set' action")?;
            let as_yaml: serde_yaml::Value = serde_yaml::from_str(
                &serde_json::to_string(v).unwrap_or_default(),
            )
            .unwrap_or(serde_yaml::Value::Null);
            if let Some(map) = cursor.as_mapping_mut() {
                map.insert(
                    serde_yaml::Value::String(last[0].to_string()),
                    as_yaml,
                );
            }
        }
        "delete" => {
            if let Some(map) = cursor.as_mapping_mut() {
                map.remove(&serde_yaml::Value::String(last[0].to_string()));
            }
        }
        other => bail!("Unknown action: {}", other),
    }
    // ⚠ Comments and formatting are lost below — load YAML grammar to avoid this.
    let out = serde_yaml::to_string(&root).context("Serializing YAML")?;
    std::fs::write(file, &out).context("Writing YAML file")?;
    Ok(format!(
        "⚠ Patched YAML '{}' at '{}' (comments may be lost — load yaml grammar to preserve)",
        file, dot_path
    ))
}

// ─── TOML ────────────────────────────────────────────────────────────────────

fn patch_toml(
    file: &str,
    path: &Path,
    action: &str,
    dot_path: &str,
    value: Option<&serde_json::Value>,
) -> Result<String> {
    // ── Surgical path ──────────────────────────────────────────────────
    if action == "set" {
        if let Ok(Some((start, end))) = TreeSitterEngine::find_node_bytes(path, dot_path) {
            let v = value.context("'value' required for 'set' action")?;
            let new_bytes = value_to_toml_scalar(v);
            let raw = std::fs::read(file).context("Reading TOML file")?;
            let patched = [&raw[..start], new_bytes.as_bytes(), &raw[end..]].concat();
            std::fs::write(file, &patched).context("Writing TOML file")?;
            return Ok(format!(
                "✅ Patched TOML '{}' at '{}' [bytes {}..{}] (comment-preserving)",
                file, dot_path, start, end
            ));
        }
    }

    // ── serde fallback (delete, or grammar not loaded) ─────────────────
    // ⚠ TOML inline comments are DESTROYED on serde round-trip.
    let raw = std::fs::read_to_string(file).context("Reading TOML file")?;
    let mut root: toml::Value = toml::from_str(&raw).context("Parsing TOML")?;
    let keys: Vec<&str> = dot_path.split('.').collect();
    let (parents, last) = keys.split_at(keys.len() - 1);
    let mut cursor = &mut root;
    for key in parents {
        cursor = cursor
            .get_mut(*key)
            .with_context(|| format!("Key '{}' not found in TOML", key))?;
    }
    match action {
        "set" => {
            let v = value.context("'value' required for 'set' action")?;
            let as_toml = json_to_toml(v);
            if let Some(tbl) = cursor.as_table_mut() {
                tbl.insert(last[0].to_string(), as_toml);
            }
        }
        "delete" => {
            if let Some(tbl) = cursor.as_table_mut() {
                tbl.remove(last[0]);
            }
        }
        other => bail!("Unknown action: {}", other),
    }
    // ⚠ Comments and formatting are lost below — load TOML grammar to avoid this.
    let out = toml::to_string_pretty(&root).context("Serializing TOML")?;
    std::fs::write(file, &out).context("Writing TOML file")?;
    Ok(format!(
        "⚠ Patched TOML '{}' at '{}' (comments may be lost — load toml grammar to preserve)",
        file, dot_path
    ))
}

// ─── Value renderers for surgical splice ─────────────────────────────────────

/// Render a JSON `Value` as a YAML scalar / inline value (no trailing newline).
fn value_to_yaml_scalar(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => {
            // Prefer bare scalars when possible; quote if special chars present.
            if needs_yaml_quoting(s) {
                format!("{:?}", s) // Rust debug quoting ≈ JSON string
            } else {
                s.clone()
            }
        }
        // For arrays/objects fall back to inline JSON (valid YAML flow syntax).
        other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
    }
}

fn needs_yaml_quoting(s: &str) -> bool {
    s.is_empty()
        || s.starts_with([' ', '\t', '{', '[', '|', '>', '&', '*', '!', '%', '@', '`'])
        || s.contains('\n')
        || matches!(
            s,
            "true" | "false" | "null" | "yes" | "no" | "on" | "off"
        )
}

/// Render a JSON `Value` as a TOML scalar (no trailing newline).
fn value_to_toml_scalar(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "\"\"".to_string(), // TOML has no `null`
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => format!("{:?}", s),
        // For arrays/objects use TOML inline table / inline array syntax.
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(value_to_toml_scalar).collect();
            format!("[{}]", items.join(", "))
        }
        serde_json::Value::Object(obj) => {
            let items: Vec<String> = obj
                .iter()
                .map(|(k, val)| format!("{k} = {}", value_to_toml_scalar(val)))
                .collect();
            format!("{{ {} }}", items.join(", "))
        }
    }
}

/// Naïve serde_json::Value → toml::Value (used by serde fallback path).
fn json_to_toml(v: &serde_json::Value) -> toml::Value {
    match v {
        serde_json::Value::Bool(b) => toml::Value::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                toml::Value::Integer(i)
            } else {
                toml::Value::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => toml::Value::String(s.clone()),
        serde_json::Value::Array(arr) => {
            toml::Value::Array(arr.iter().map(json_to_toml).collect())
        }
        serde_json::Value::Object(obj) => {
            let mut tbl = toml::map::Map::new();
            for (k, val) in obj {
                tbl.insert(k.clone(), json_to_toml(val));
            }
            toml::Value::Table(tbl)
        }
        serde_json::Value::Null => toml::Value::String("null".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn test_patch_json_nested() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        fs::write(path, r#"{"db": {"host": "localhost", "port": 5432}}"#).unwrap();

        patch_json(path, file.path(), "set", "db.port", Some(&json!(5433))).unwrap();
        let val: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(val["db"]["port"], 5433);

        patch_json(path, file.path(), "delete", "db.host", None).unwrap();
        let val2: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert!(val2["db"]["host"].is_null());
    }

    #[test]
    fn test_patch_json_array() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        fs::write(path, r#"{"features": ["a", "b"]}"#).unwrap();

        patch_json(
            path,
            file.path(),
            "set",
            "features",
            Some(&json!(["a", "b", "c"])),
        )
        .unwrap();
        let val: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(val["features"][2], "c");
    }
}
