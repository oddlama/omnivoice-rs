use std::{
    collections::{HashMap, HashSet},
    sync::OnceLock,
};

const LANG_ID_NAME_MAP_TSV: &str = include_str!("lang_id_name_map.tsv");

static LANG_NAME_TO_ID: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
static LANG_IDS: OnceLock<HashSet<&'static str>> = OnceLock::new();

pub fn resolve_language(language: Option<&str>) -> Option<String> {
    let language = language?.trim();
    if language.is_empty() || language.eq_ignore_ascii_case("none") {
        return None;
    }
    if lang_ids().contains(language) {
        return Some(language.to_string());
    }

    let normalized = language.to_ascii_lowercase();
    lang_name_to_id()
        .get(normalized.as_str())
        .map(|value| (*value).to_string())
}

fn lang_name_to_id() -> &'static HashMap<&'static str, &'static str> {
    LANG_NAME_TO_ID.get_or_init(|| {
        let mut mapping = HashMap::with_capacity(700);
        for line in LANG_ID_NAME_MAP_TSV
            .lines()
            .skip(1)
            .filter(|line| !line.is_empty())
        {
            let mut fields = line.split('\t');
            let Some(language_id) = fields.next() else {
                continue;
            };
            let Some(language_name) = fields.next() else {
                continue;
            };
            let normalized_name: &'static str =
                Box::leak(language_name.to_ascii_lowercase().into_boxed_str());
            let normalized_id: &'static str = Box::leak(language_id.to_string().into_boxed_str());
            mapping.insert(normalized_name, normalized_id);
        }
        mapping
    })
}

fn lang_ids() -> &'static HashSet<&'static str> {
    LANG_IDS.get_or_init(|| {
        let mut ids = HashSet::with_capacity(700);
        for line in LANG_ID_NAME_MAP_TSV
            .lines()
            .skip(1)
            .filter(|line| !line.is_empty())
        {
            if let Some(language_id) = line.split('\t').next() {
                let normalized_id: &'static str =
                    Box::leak(language_id.to_string().into_boxed_str());
                ids.insert(normalized_id);
            }
        }
        ids
    })
}
