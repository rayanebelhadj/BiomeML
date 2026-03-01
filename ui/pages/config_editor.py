import streamlit as st
import yaml
from components.experiment_io import CONFIG_YAML, EXPERIMENTS_YAML

st.header("Configuration")


def _load_text(path):
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def _save_yaml(path, text):
    """Validate YAML then write. Returns True on success."""
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as e:
        st.error(f"Invalid YAML: {e}")
        return False
    path.write_text(text)
    return True


tab_config, tab_exp = st.tabs(["config.yaml", "experiments.yaml"])

with tab_config:
    config_text = _load_text(CONFIG_YAML)
    edited = st.text_area(
        "config.yaml",
        value=config_text,
        height=550,
        key="config_editor",
        label_visibility="collapsed",
    )
    if st.button("Save config.yaml", key="save_config"):
        if _save_yaml(CONFIG_YAML, edited):
            st.success("Saved config.yaml")

with tab_exp:
    exp_text = _load_text(EXPERIMENTS_YAML)
    edited_exp = st.text_area(
        "experiments.yaml",
        value=exp_text,
        height=550,
        key="exp_editor",
        label_visibility="collapsed",
    )
    if st.button("Save experiments.yaml", key="save_exp"):
        if _save_yaml(EXPERIMENTS_YAML, edited_exp):
            st.success("Saved experiments.yaml")
