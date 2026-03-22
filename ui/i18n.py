"""Internationalization (i18n) framework for the Streamlit UI.

Provides a singleton ``I18n`` class that loads JSON translation files from
``ui/locales/`` and exposes a ``t(key, **kwargs)`` lookup with automatic
English fallback.

Usage::

    from ui.i18n import t

    st.subheader(t("overview.gpu_info"))
    st.progress(pct, text=t("overview.vram_usage", used="4.2", total="32.0"))
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_UI_LANGS: dict[str, str] = {
    "en": "English",
    "ko": "한국어",
    "ja": "日本語",
    "zh": "中文",
    "fr": "Français",
    "es": "Español",
    "de": "Deutsch",
    "pt": "Português",
}

_LOCALES_DIR = Path(__file__).parent / "locales"


class I18n:
    """Singleton translation manager.

    Loads ``{lang}.json`` files from ``ui/locales/`` at first access.
    Lookups fall back to English, then return the raw key if missing.
    """

    _instance: "I18n | None" = None

    def __init__(self) -> None:
        self._translations: dict[str, dict[str, str]] = {}
        self._lang = "en"
        self._load_all()

    @classmethod
    def get(cls) -> "I18n":
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def lang(self) -> str:
        """Current UI language code."""
        return self._lang

    def set_language(self, lang: str) -> None:
        """Set the active language (falls back to ``en``)."""
        self._lang = lang if lang in self._translations else "en"

    def t(self, key: str, **kwargs: object) -> str:
        """Look up *key* in the current language.

        Falls back to English, then to the key itself.  Keyword
        arguments are interpolated via :meth:`str.format`.
        """
        text = self._translations.get(self._lang, {}).get(key)
        if text is None:
            text = self._translations.get("en", {}).get(key, key)
        return text.format(**kwargs) if kwargs else text

    def _load_all(self) -> None:
        """Load every ``*.json`` file in the locales directory."""
        if not _LOCALES_DIR.is_dir():
            logger.warning("Locales directory not found: %s", _LOCALES_DIR)
            return
        for json_file in sorted(_LOCALES_DIR.glob("*.json")):
            lang = json_file.stem
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                self._translations[lang] = data
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load locale: %s", json_file)


def get_i18n() -> I18n:
    """Return the singleton :class:`I18n` instance."""
    return I18n.get()


def t(key: str, **kwargs: object) -> str:
    """Module-level shortcut for ``I18n.get().t(key, **kwargs)``."""
    return I18n.get().t(key, **kwargs)
