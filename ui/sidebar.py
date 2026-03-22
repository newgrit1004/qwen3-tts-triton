"""Sidebar controls for the Streamlit dashboard.

Renders UI language selector only. TTS controls are in the Playground tab.
"""

import streamlit as st

from ui.i18n import SUPPORTED_UI_LANGS, get_i18n, t


def render_sidebar() -> None:
    """Render minimal sidebar with UI language selector."""
    st.sidebar.title(t("sidebar.title"))
    ui_lang = st.sidebar.selectbox(
        "\U0001f310 UI Language",
        list(SUPPORTED_UI_LANGS.keys()),
        format_func=lambda x: SUPPORTED_UI_LANGS[x],
        key="ui_lang_select",
    )
    get_i18n().set_language(ui_lang)
