"""The bot personalities. Each bot has a name, a personality (its system prompt) and a color for the graph.
Which folders feed a bot's private memory is kept with the watched folders (daemons/watcher.py)."""
import json
import os
import re
import threading

BOTS_PATH = "config/bots.json"
SHARED = "shared"  # memory partition every bot (and the main Nuero AI) can search

# Distinct from the graph's own colors (blue/green/pink files, gold highlight)
BOT_COLORS = ["#f97316", "#a3e635", "#22d3ee", "#818cf8", "#c084fc", "#fb7185",
              "#14b8a6", "#e879f9", "#b45309", "#ef4444", "#cbd5e1", "#f8fafc"]

# Files in a bot's folder that probably describe the bot, used to pre-fill its personality
_PERSONA_FILE = re.compile(r"^(persona|personality|identity|instructions?|system[ _-]?prompt|character|profile)\b.*\.(txt|md)$", re.I)
_PERSONA_MAX_CHARS = 6000


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "bot"


def find_persona_text(folder):
    """Personality text from a persona-like file at the top of the folder, or ''."""
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return ""
    for name in names:
        path = os.path.join(folder, name)
        if _PERSONA_FILE.match(name) and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read(_PERSONA_MAX_CHARS).strip()
            except OSError:
                continue
    return ""


class BotRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        try:
            with open(BOTS_PATH, "r", encoding="utf-8") as f:
                self.bots = json.load(f).get("bots", [])
        except (OSError, ValueError):
            self.bots = []

    def _save(self):
        tmp = BOTS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"bots": self.bots}, f, indent=2)
        os.replace(tmp, BOTS_PATH)

    def get(self, bot_id):
        return next((b for b in self.bots if b["id"] == bot_id), None)

    def require(self, bot_id):
        bot = self.get(bot_id)
        if not bot:
            raise ValueError(f"No Family Node called '{bot_id}'.")
        return bot

    def find(self, hint):
        """Bot by id or (partial, case-insensitive) name."""
        hint = hint.strip().lower()
        exact = [b for b in self.bots if hint in (b["id"], b["name"].lower())]
        partial = [b for b in self.bots if hint in b["name"].lower()]
        return (exact or partial or [None])[0]

    def add(self, name, personality=""):
        name = name.strip()
        if not name:
            raise ValueError("Give the Family Node a name.")
        with self.lock:
            if any(b["name"].lower() == name.lower() for b in self.bots):
                raise ValueError(f"There's already a Family Node called {name}.")
            base, n = _slug(name), 2
            bot_id = base
            while bot_id == SHARED or self.get(bot_id):
                bot_id, n = f"{base}-{n}", n + 1
            used = {b["color"] for b in self.bots}
            color = next((c for c in BOT_COLORS if c not in used), BOT_COLORS[len(self.bots) % len(BOT_COLORS)])
            bot = {"id": bot_id, "name": name, "personality": personality.strip(), "color": color}
            self.bots.append(bot)
            self._save()
        return bot

    def update(self, bot_id, name=None, personality=None, images=None, music=None):
        """images / music: whether the bot may create images / share songs on its own (default True when unset)."""
        with self.lock:
            bot = self.require(bot_id)
            if images is not None:
                bot["images"] = bool(images)
            if music is not None:
                bot["music"] = bool(music)
            if name is not None and name.strip():
                if any(b is not bot and b["name"].lower() == name.strip().lower() for b in self.bots):
                    raise ValueError(f"There's already a Family Node called {name.strip()}.")
                bot["name"] = name.strip()
            if personality is not None:
                bot["personality"] = personality.strip()
            self._save()
        return bot

    # ---- skills: named instruction sets a bot can switch between in a conversation ----

    def add_skill(self, bot_id, name, instructions):
        name, instructions = (name or "").strip(), (instructions or "").strip()
        if not name or not instructions:
            raise ValueError("A skill needs a name and instructions.")
        with self.lock:
            bot = self.require(bot_id)
            skills = bot.setdefault("skills", [])
            if any(s["name"].lower() == name.lower() for s in skills):
                raise ValueError(f"{bot['name']} already has a skill called {name}.")
            base, n = _slug(name), 2
            skill_id = base
            while any(s["id"] == skill_id for s in skills):
                skill_id, n = f"{base}-{n}", n + 1
            skill = {"id": skill_id, "name": name[:60], "instructions": instructions[:20000]}
            skills.append(skill)
            self._save()
        return skill

    def update_skill(self, bot_id, skill_id, name=None, instructions=None):
        with self.lock:
            skill = self._skill(bot_id, skill_id)
            if name is not None and name.strip():
                skill["name"] = name.strip()[:60]
            if instructions is not None and instructions.strip():
                skill["instructions"] = instructions.strip()[:20000]
            self._save()
        return skill

    def delete_skill(self, bot_id, skill_id):
        with self.lock:
            bot = self.require(bot_id)
            self._skill(bot_id, skill_id)
            bot["skills"] = [s for s in bot.get("skills", []) if s["id"] != skill_id]
            self._save()

    def _skill(self, bot_id, skill_id):
        skill = next((s for s in self.require(bot_id).get("skills", []) if s["id"] == skill_id), None)
        if not skill:
            raise ValueError("That skill doesn't exist any more.")
        return skill

    def skill(self, bot_id, skill_id):
        """The skill, or None if the bot or skill is gone."""
        bot = self.get(bot_id)
        return next((s for s in bot.get("skills", []) if s["id"] == skill_id), None) if bot else None

    def delete(self, bot_id):
        with self.lock:
            self.require(bot_id)
            self.bots = [b for b in self.bots if b["id"] != bot_id]
            self._save()


_registry = None
_registry_lock = threading.Lock()


def get_bots():
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = BotRegistry()
        return _registry
