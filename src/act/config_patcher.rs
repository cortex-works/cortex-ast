use anyhow::{bail, Context, Result};
use std::path::Path;

/// Patch a single key in a JSON, YAML, or TOML config file using dot-path notation.
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
        "json" => patch_json(file, action, dot_path, value),
        "yaml" | "yml" => patch_yaml(file, action, dot_path, value),
        "toml" => patch_toml(file, action, dot_path, value),
        other => bail!("Unsupported config file extension: .{}", other),
    }
}

// ─── JSON ────────────────────────────────────────────────────────────────────

fn patch_json(
    file: &str,
    action: &str,
    dot_path: &str,
    value: Option<&serde_json::Value>,
) -> Result<String> {
    let raw = std::fs::read_to_string(file).context("Reading JSON file")?;
    let mut root: serde_json::Value =
        serde_json::from_str(&raw).context("Parsing JSON")?;

    let keys: Vec<&str> = dot_path.split('.').collect();
    let (parents, last) = keys.split_at(keys.len() - 1);

    // Navigate to the parent object
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
    action: &str,
    dot_path: &str,
    value: Option<&serde_json::Value>,
) -> Result<String> {
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
            // Convert serde_json::Value -> serde_yaml::Value via JSON string round-trip
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

    let out = serde_yaml::to_string(&root).context("Serializing YAML")?;
    std::fs::write(file, &out).context("Writing YAML file")?;
    Ok(format!("✅ Patched YAML '{}' at '{}'", file, dot_path))
}

// ─── TOML ────────────────────────────────────────────────────────────────────

fn patch_toml(
    file: &str,
    action: &str,
    dot_path: &str,
    value: Option<&serde_json::Value>,
) -> Result<String> {
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

    let out = toml::to_string_pretty(&root).context("Serializing TOML")?;
    std::fs::write(file, &out).context("Writing TOML file")?;
    Ok(format!("✅ Patched TOML '{}' at '{}'", file, dot_path))
}

/// Naïve serde_json::Value → toml::Value conversion.
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

        patch_json(path, "set", "db.port", Some(&json!(5433))).unwrap();
        let val: serde_json::Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(val["db"]["port"], 5433);

        patch_json(path, "delete", "db.host", None).unwrap();
        let val2: serde_json::Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert!(val2["db"]["host"].is_null());
    }

    #[test]
    fn test_patch_json_array() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        fs::write(path, r#"{"features": ["a", "b"]}"#).unwrap();

        patch_json(path, "set", "features", Some(&json!(["a", "b", "c"]))).unwrap();
        let val: serde_json::Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(val["features"][2], "c");
    }
}
