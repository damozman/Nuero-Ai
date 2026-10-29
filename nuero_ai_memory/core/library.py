"""The library: books (collections) of saved items the user curates, plus one archive book per bot
where it keeps what it decides is important. Items are copies, so a book stays intact even if the
memory or file it came from changes or is forgotten. Stored in storage/library.json."""
import datetime
import json
import os
import re
import threading
import uuid

LIBRARY_PATH = "storage/library.json"
CONTEXT_BUDGET = 30000  # characters of attached books sent with each message
FILE_MAX_CHARS = 60000  # a whole file added to a book
TEXT_EXTENSIONS = ('.txt', '.md', '.log', '.py')
BOOK_COLORS = ["#7c3aed", "#0891b2", "#b45309", "#be123c", "#15803d", "#1d4ed8", "#a21caf", "#c2410c", "#0f766e", "#4d7c0f"]

# Bots add this line when something feels worth keeping
ARCHIVE_MARKER = re.compile(r"\[\[\s*ARCHIVE\s*:\s*(.+?)\s*\|\s*(.+?)\s*\]\]", re.I | re.S)
ARCHIVE_NOTE = (
    "You have a personal archive. If something in this conversation feels genuinely important to you and "
    "worth keeping long-term, you may archive it by ending your reply with one line in exactly this form: "
    "[[ARCHIVE: short title | what you want to keep, written out in full]]. Use it sparingly, at most once per "
    "reply, and never mention or explain this syntax."
)


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


class Library:
    def __init__(self):
        self.lock = threading.Lock()
        try:
            with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
                self.books = json.load(f).get("books", [])
        except (OSError, ValueError):
            self.books = []
        self._ensure_guide()

    def _ensure_guide(self):
        """Keeps the built-in guide book present and up to date (it holds no personal data)."""
        from core.guide import GUIDE_PAGES, GUIDE_TITLE, GUIDE_VERSION
        guide = next((b for b in self.books if b["kind"] == "guide"), None)
        if guide and guide.get("version") == GUIDE_VERSION:
            return
        if not guide:
            guide = {"id": "guide", "title": GUIDE_TITLE, "emoji": "📘", "color": "#0369a1", "kind": "guide",
                     "bot": None, "created": _now()}
            self.books.insert(0, guide)
        guide.update(version=GUIDE_VERSION, updated=_now(), items=[
            {"id": f"guide-{n}", "title": title, "text": text, "source": "", "origin": "Built-in guide",
             "bot": None, "image": None, "music": None, "added": _now()}
            for n, (title, text) in enumerate(GUIDE_PAGES)
        ])
        try:
            self._save()
        except OSError:
            pass  # storage not writable yet; the guide is still available in memory

    def _save(self):
        tmp = LIBRARY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"books": self.books}, f, indent=2)
        os.replace(tmp, LIBRARY_PATH)

    def get(self, book_id):
        return next((b for b in self.books if b["id"] == book_id), None)

    def require(self, book_id):
        book = self.get(book_id)
        if not book:
            raise ValueError("That book doesn't exist any more.")
        return book

    def summaries(self):
        return [
            {k: v for k, v in b.items() if k != "items"}
            | {"count": len(b["items"]), "images": sum(1 for i in b["items"] if i.get("image")),
               "songs": sum(1 for i in b["items"] if i.get("music"))}
            for b in self.books
        ]

    def create_book(self, title, emoji="📘", color=None, kind="user", bot=None):
        title = (title or "").strip()
        if not title:
            raise ValueError("Give the book a title.")
        with self.lock:
            book = {
                "id": uuid.uuid4().hex[:10], "title": title[:80], "emoji": (emoji or "📘").strip()[:4] or "📘",
                "color": color or BOOK_COLORS[len(self.books) % len(BOOK_COLORS)],
                "kind": kind, "bot": bot, "created": _now(), "updated": _now(), "items": [],
            }
            self.books.append(book)
            self._save()
        return book

    def update_book(self, book_id, title=None, emoji=None, color=None):
        with self.lock:
            book = self.require(book_id)
            if title is not None and title.strip():
                book["title"] = title.strip()[:80]
            if emoji is not None and emoji.strip():
                book["emoji"] = emoji.strip()[:4]
            if color is not None and re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                book["color"] = color
            book["updated"] = _now()
            self._save()
        return book

    def delete_book(self, book_id):
        with self.lock:
            if self.require(book_id)["kind"] == "guide":
                raise ValueError("The built-in guide can't be deleted (it's simply not used unless attached).")
            self.books = [b for b in self.books if b["id"] != book_id]
            self._save()

    def add_item(self, book_id, text="", title="", source="", origin="", bot=None, image=None, music=None):
        """origin: where it came from, e.g. "Athena's reply", "Note", "Memory", "File".
        music: a song card (title, artist, album, artwork, preview, links) from core.music."""
        text, title = (text or "").strip(), (title or "").strip()
        if not text and not image and not music:
            raise ValueError("Nothing to add.")
        if music:
            music = {k: music.get(k) for k in ("title", "artist", "album", "artwork", "preview", "links")}
            title = title or f"{music['title']} — {music['artist']}".strip(" —")
        with self.lock:
            book = self.require(book_id)
            item = {
                "id": uuid.uuid4().hex[:10], "title": title[:120] or (text.split("\n", 1)[0][:60] if text else "Image"),
                "text": text, "source": source, "origin": origin, "bot": bot, "image": image, "music": music,
                "added": _now(),
            }
            book["items"].append(item)
            book["updated"] = _now()
            self._save()
        return item

    def add_file(self, book_id, path, origin="File"):
        path = os.path.abspath(path.strip().strip('"'))
        if not path.lower().endswith(TEXT_EXTENSIONS) or not os.path.isfile(path):
            raise ValueError("Only .txt, .md, .log and .py files can be added.")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(FILE_MAX_CHARS + 1)
        if len(text) > FILE_MAX_CHARS:
            text = text[:FILE_MAX_CHARS] + "\n[… rest of the file not included]"
        return self.add_item(book_id, text=text, title=os.path.basename(path), source=path, origin=origin)

    def remove_item(self, book_id, item_id):
        with self.lock:
            book = self.require(book_id)
            book["items"] = [i for i in book["items"] if i["id"] != item_id]
            book["updated"] = _now()
            self._save()

    def archive_for(self, bot_id, bot_name, color=None):
        """The bot's own archive book, created the first time it archives something."""
        book = next((b for b in self.books if b["kind"] == "archive" and b["bot"] == bot_id), None)
        return book or self.create_book(f"{bot_name}'s Archive", "🗄", color, kind="archive", bot=bot_id)

    def playlist_for(self, bot_id, bot_name, color=None):
        """The bot's playlist book, where every song it shares is collected."""
        book = next((b for b in self.books if b["kind"] == "playlist" and b["bot"] == bot_id), None)
        return book or self.create_book(f"{bot_name}'s Playlist", "🎵", color, kind="playlist", bot=bot_id)

    def context_for(self, book_ids):
        """Attached books as text for the AI, newest items first within the size budget."""
        parts, used, cut = [], 0, False
        for book in filter(None, (self.get(b) for b in book_ids or [])):
            lines = [f"### Book: {book['title']}"]
            if book["kind"] == "guide":
                lines.append("(How the Nuero AI app itself works. Answer questions about the app only from this guide: "
                             "don't invent buttons, icons or steps it doesn't describe. If something isn't covered, "
                             "say you're not sure.)")
            # The guide reads best in order; other books newest first
            items = book["items"] if book["kind"] == "guide" else list(reversed(book["items"]))
            for item in items:
                entry = f"- {item['title']}"
                if item.get("image"):
                    entry += " (an image)"
                if item.get("music"):
                    entry += " (a song)"
                if item.get("text") and item["text"] != item["title"]:
                    entry += f": {item['text']}"
                if used + len(entry) > CONTEXT_BUDGET:
                    cut = True
                    break
                lines.append(entry)
                used += len(entry)
            parts.append("\n".join(lines))
        if not parts:
            return ""
        note = "\n(Some older pages were left out to keep this short.)" if cut else ""
        return "\n\n".join(parts) + note


_library = None
_library_lock = threading.Lock()


def get_library():
    global _library
    with _library_lock:
        if _library is None:
            _library = Library()
        return _library
