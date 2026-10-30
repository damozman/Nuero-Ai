"""Household settings shown throughout the app: what the system is called and the human's name.
Kept in config/profile.json; every part of the app reads the system name from here."""
import json
import os
import threading

PROFILE_PATH = "config/profile.json"
DEFAULT_SYSTEM_NAME = "Neo AI"
LEGACY_NAMES = ("Nuero AI",)  # earlier names, swapped for the current one in older text and files

_lock = threading.Lock()


def load():
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save(**changes):
    """Updates the given settings (strings), keeping the others."""
    with _lock:
        data = load()
        for key, value in changes.items():
            if value is not None:
                data[key] = str(value).strip()[:60]
        tmp = PROFILE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, PROFILE_PATH)
    return data


def system_name():
    return load().get("system_name", "").strip() or DEFAULT_SYSTEM_NAME


def user_name():
    return load().get("user_name", "").strip()


def with_system_name(text, name=None):
    """Fills {system_name} and replaces earlier names of the app in text written before a rename."""
    name = name or system_name()
    for old in LEGACY_NAMES:
        text = text.replace(old, name)
    return text.replace("{system_name}", name)
