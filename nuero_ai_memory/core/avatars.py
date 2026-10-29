"""Profile pictures for bots and the main Nuero AI ("main"), kept in storage/avatars and served at /avatars."""
import base64
import json
import os
import re
import threading
import time

AVATARS_DIR = "storage/avatars"
INDEX_PATH = "storage/avatars/avatars.json"  # owner id -> file name
MAX_BYTES = 4 * 1024 * 1024
_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}

_lock = threading.Lock()


def _index():
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_index(index):
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, INDEX_PATH)


def avatar_url(owner):
    name = _index().get(owner)
    return f"/avatars/{name}" if name else None


def _store(owner, data, ext):
    """Writes the new picture under a fresh name (so browsers don't show a cached old one)."""
    os.makedirs(AVATARS_DIR, exist_ok=True)
    safe = re.sub(r"[^a-z0-9-]", "", owner.lower()) or "bot"
    name = f"{safe}-{int(time.time())}{ext}"
    with _lock:
        index = _index()
        old = index.get(owner)
        with open(os.path.join(AVATARS_DIR, name), "wb") as f:
            f.write(data)
        index[owner] = name
        _save_index(index)
        if old and old != name:
            try:
                os.remove(os.path.join(AVATARS_DIR, old))
            except OSError:
                pass
    return f"/avatars/{name}"


def save_data_url(owner, data_url):
    """An uploaded picture as a data: URL (the page shrinks it to a square first)."""
    match = re.fullmatch(r"data:(image/[a-z+]+);base64,(.+)", (data_url or "").strip(), re.S)
    if not match or match.group(1) not in _TYPES:
        raise ValueError("Use a PNG, JPG, WebP or GIF picture.")
    data = base64.b64decode(match.group(2))
    if len(data) > MAX_BYTES:
        raise ValueError("That picture is too large (max 4 MB).")
    return _store(owner, data, _TYPES[match.group(1)])


def save_file(owner, path):
    """Uses an existing image file (e.g. one the image AI just created) as the avatar."""
    with open(path, "rb") as f:
        data = f.read()
    return _store(owner, data, os.path.splitext(path)[1].lower() or ".png")


def remove(owner):
    with _lock:
        index = _index()
        name = index.pop(owner, None)
        _save_index(index)
    if name:
        try:
            os.remove(os.path.join(AVATARS_DIR, name))
        except OSError:
            pass


def all_urls():
    return {owner: f"/avatars/{name}" for owner, name in _index().items()}
