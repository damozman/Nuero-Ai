import datetime
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.bots import get_bots, SHARED
from core.images import get_studio, IMAGE_MARKER, AUTO_IMAGE_NOTE
from core.library import get_library, ARCHIVE_MARKER, ARCHIVE_NOTE
from core import music
from core.music import SONG_MARKER, SONG_NOTE, MAX_SONGS
from core.providers import PROVIDERS, ProviderError
from core.scrubber import MemoryScrubber
from core.vector_engine import get_vault

SETTINGS_PATH = "config/ai_settings.json"  # chosen AI and the last model used with each
# How many council members are asked at once; lower it if the AI service rate-limits you
COUNCIL_PARALLEL = int(os.environ.get("NUERO_COUNCIL_PARALLEL", "12"))
HISTORY_TURNS = 10  # recent messages of the same conversation sent along for context
HISTORY_CHARS = 2000  # per message, so one long reply can't crowd out the rest
COUNCIL_NOTE = ("You are one member of a council of assistants answering the same message. "
                "Give your own perspective, drawing on your own memories. Keep it focused.")

_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    """The terminal chat and the dashboard share one pipeline, so switching AI in one switches both."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = MemoryPipeline()
        return _pipeline


class MemoryPipeline:
    def __init__(self):
        self.scrubber = MemoryScrubber()
        self.vault = get_vault()
        self.bots = get_bots()
        self.lock = threading.Lock()
        settings = self._load_settings()
        self.models = settings.get("models", {})  # provider id -> last chosen model
        self.provider_id = self._initial_provider(settings.get("provider"))

    # ---- choosing the AI ----

    @staticmethod
    def _load_settings():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_settings(self):
        tmp = SETTINGS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"provider": self.provider_id, "models": self.models}, f, indent=2)
        os.replace(tmp, SETTINGS_PATH)

    @staticmethod
    def _initial_provider(saved):
        """NUERO_PROVIDER (set by the launcher for this session), then the saved choice, then whichever
        service is available."""
        for candidate in (os.environ.get("NUERO_PROVIDER", "").strip().lower(), saved):
            if candidate in PROVIDERS and PROVIDERS[candidate].has_key():
                return candidate
        available = [pid for pid, p in PROVIDERS.items() if p.has_key()]
        return available[0] if available else "gemini"

    @property
    def provider(self):
        return PROVIDERS[self.provider_id]

    @property
    def model(self):
        # Ollama has no fixed default: fall back to the first model installed
        return (self.models.get(self.provider_id) or self.provider.default_model
                or self.provider.models()[0]["id"])

    def describe(self):
        return f"{self.provider.label} ({self.model})"

    def set_ai(self, provider_id, model_hint=None):
        """Switch service and/or model. model_hint may be a full id or a fragment like 'sonnet'."""
        provider_id = (provider_id or self.provider_id).lower()
        if provider_id not in PROVIDERS:
            raise ProviderError(f"Unknown AI '{provider_id}'. Choose from: {', '.join(PROVIDERS)}.")
        provider = PROVIDERS[provider_id]
        if not provider.has_key():
            raise ProviderError(provider.missing_hint)
        model = None
        if model_hint:
            ids = [m["id"] for m in provider.models()]
            hint = model_hint.strip().lower()
            matches = [m for m in ids if m.lower() == hint] or [m for m in ids if hint in m.lower()]
            if not matches:
                raise ProviderError(f"{provider.label} has no model matching '{model_hint}'. Options: {', '.join(ids)}")
            model = matches[0]
        with self.lock:
            self.provider_id = provider_id
            if model:
                self.models[provider_id] = model
            self._save_settings()
        return self.describe()

    def ai_status(self):
        return {
            "current": {"provider": self.provider_id, "model": self.model},
            "providers": [
                {
                    "id": p.id,
                    "label": p.label,
                    "key_env": p.key_env,
                    "available": p.has_key(),
                    "local": p.id == "ollama",
                    "missing_label": p.missing_label,
                    "missing_hint": p.missing_hint,
                    "model": self.models.get(p.id) or p.default_model,
                    "models": p.models() if p.has_key() else [],
                }
                for p in PROVIDERS.values()
            ],
        }

    # ---- chatting ----

    def _get_identity_frequency(self) -> str:
        path = "config/identity_frequency.txt"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return "You are an advanced, personal, responsive AI core assistant."

    def chat(self, user_input: str, bot_id=None, history=None) -> str:
        return self.respond(user_input, bot_id, history=history)["reply"]

    def _clean_history(self, history):
        """Last few turns of this conversation, trimmed and masked like everything else that leaves the machine."""
        turns = [t for t in history or [] if t.get("role") in ("user", "assistant") and str(t.get("text", "")).strip()]
        return [{"role": t["role"], "text": self.scrubber.mask(str(t["text"])[:HISTORY_CHARS])}
                for t in turns[-HISTORY_TURNS:]]

    def respond(self, user_input: str, bot_id=None, council=False, history=None, books=None, skill=None) -> dict:
        """One reply. bot_id=None is the main Nuero AI (shared memories); a bot searches its own + shared.
        history: earlier turns of this same conversation, [{"role": "user"|"assistant", "text": ...}].
        books: library book ids attached to this conversation; the AI reads them in full.
        skill: id of one of the bot's skills to use for this reply.
        Returns the reply, the memories it drew on, and who answered (for the dashboard)."""
        if bot_id:
            bot = self.bots.get(bot_id)
            if not bot:
                raise ProviderError(f"No bot called '{bot_id}'.")
            identity = bot["personality"] or f"You are {bot['name']}, a personal AI assistant."
            name = bot["name"]
            partitions = [bot_id, SHARED]
        else:
            identity, name, partitions = self._get_identity_frequency(), "Nuero AI", [SHARED]
        active_skill = self.bots.skill(bot_id, skill) if bot_id and skill else None
        if active_skill:
            identity = (f"{identity}\n\n[ACTIVE SKILL: {active_skill['name']}]\n"
                        f"Apply this skill in your reply, staying in character:\n{active_skill['instructions']}")
        if council:
            identity = f"{identity}\n\n{COUNCIL_NOTE}"
        # Bots may create images on their own, except in Council (one question could trigger a dozen)
        # and on the local AI, where the point is that nothing leaves the computer
        studio = get_studio()
        may_draw = not council and self.provider_id != "ollama" and studio.can_auto() and (
            bot.get("images", True) if bot_id else studio.settings["main_auto"])
        if may_draw:
            identity = f"{identity}\n\n{AUTO_IMAGE_NOTE}"
        identity = f"{identity}\n\n{ARCHIVE_NOTE}"
        may_share_music = bot.get("music", True) if bot_id else True
        if may_share_music:
            identity = f"{identity}\n\n{SONG_NOTE}"

        memories = self.vault.search_memories(user_input, limit=3, partitions=partitions)
        raw_context = "\n---\n".join([m.payload["text"] for m in memories])
        masked_context = self.scrubber.mask(raw_context)
        masked_prompt = self.scrubber.mask(user_input)
        identity_frequency = self.scrubber.mask(identity)

        system_instruction = f"""
        {identity_frequency}

        [SECURE RETRIEVED MEMORY MATRIX]:
        {masked_context}
        """
        library_context = get_library().context_for(books)
        if library_context:
            system_instruction += (
                "\n[REFERENCE LIBRARY - books the user attached to this conversation; draw on them]:\n"
                + self.scrubber.mask(library_context) + "\n"
            )

        # Snapshot the choice so a switch mid-request can't mix providers
        provider, model = self.provider, self.model
        reply = provider.generate(model, masked_prompt, system_instruction, self._clean_history(history))

        # Pull out any image the bot decided to create; the marker never reaches the chat
        image_prompts = IMAGE_MARKER.findall(reply)
        reply = IMAGE_MARKER.sub("", reply).strip()
        images, image_error = [], None
        if image_prompts and may_draw:
            try:
                images.append(self._make_image(image_prompts[0], bot_id, name, requested=False))
            except Exception as e:
                image_error = f"{name} tried to create an image, but it failed: {e}"

        # Things the bot chose to keep go into its own archive book (and its memory)
        archived = []
        for title, text in ARCHIVE_MARKER.findall(reply)[:1]:
            archived.append(self._archive(bot_id, name, self.scrubber.unmask(title), self.scrubber.unmask(text)))
        reply = ARCHIVE_MARKER.sub("", reply).strip()

        # Songs the bot shared: looked up, collected in its playlist book, and remembered
        song_marks = SONG_MARKER.findall(reply)[:MAX_SONGS]
        reply = SONG_MARKER.sub("", reply).strip()
        songs = self._share_songs(song_marks, bot_id, name) if song_marks and may_share_music else []

        return {
            "songs": songs,
            "reply": self.scrubber.unmask(reply),
            "images": images,
            "image_error": image_error,
            "archived": archived,
            "sources": [m.id for m in memories],
            "memories": [
                {
                    "id": m.id,
                    "text": m.payload.get("text", ""),
                    "source": str(m.payload.get("metadata", {}).get("source", "")).split(":", 1)[-1] or "Saved from chat",
                    "bot": m.payload.get("metadata", {}).get("bot_id", SHARED),
                }
                for m in memories
            ],
            "ai": f"{provider.label} · {model}",
            "skill": active_skill["name"] if active_skill else None,
            "bot": bot_id,
            "bot_name": name,
        }

    def _share_songs(self, marks, bot_id, name):
        queries = [(self.scrubber.unmask(q), self.scrubber.unmask(why or "")) for q, why in marks]
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            cards = list(pool.map(lambda qw: music.lookup(qw[0]), queries))
        bot = self.bots.get(bot_id) if bot_id else None
        playlist = get_library().playlist_for(bot_id or "main", name, bot["color"] if bot else "#0096ff")
        shared = []
        for card, (_, why) in zip(cards, queries):
            get_library().add_item(playlist["id"], text=why, origin=f"Shared by {name}", bot=bot_id, music=card)
            self.vault.add_memory(
                f"{name} shared the song \"{card['title']}\" by {card['artist']} on {datetime.date.today():%B %d, %Y}"
                + (f" because: {why}" if why else "."),
                metadata={"origin": "song", "bot_id": bot_id or SHARED},
            )
            shared.append({**card, "why": why, "by": name, "playlist": playlist["title"]})
        print(f"\n[MUSIC]: {name} shared {', '.join(c['title'] for c in cards)}")
        return shared

    def _archive(self, bot_id, name, title, text):
        bot = self.bots.get(bot_id) if bot_id else None
        book = get_library().archive_for(bot_id or "main", name, bot["color"] if bot else "#0096ff")
        get_library().add_item(book["id"], text=text, title=title, origin=f"Archived by {name}", bot=bot_id)
        self.vault.add_memory(
            f"{name} archived this on {datetime.date.today():%B %d, %Y} as \"{title}\": {text}",
            metadata={"origin": "archive", "bot_id": bot_id or SHARED},
        )
        print(f"\n[LIBRARY]: {name} archived \"{title}\"")
        return {"book": book["id"], "book_title": book["title"], "title": title}

    def _make_image(self, masked_prompt, bot_id, name, requested):
        """Creates an image and remembers it in the owner's memory."""
        # Mask again right before it leaves the machine: a bot's own prompt could still contain private details
        masked_prompt = self.scrubber.mask(masked_prompt)
        info = get_studio().create(masked_prompt, auto=not requested)
        prompt = self.scrubber.unmask(masked_prompt)
        how = "at the user's request" if requested else "on its own initiative"
        self.vault.add_memory(
            f"Image created on {datetime.date.today():%B %d, %Y} by {name} {how} (file {info['file']}). "
            f"It shows: {prompt}",
            metadata={"origin": "image", "bot_id": bot_id or SHARED, "image": info["file"]},
        )
        return {**info, "prompt": prompt, "by": name, "requested": requested}

    def create_image(self, prompt: str, bot_id=None) -> dict:
        """An image the user asked for, saved to the memory of whoever they're talking to."""
        if bot_id and not self.bots.get(bot_id):
            raise ProviderError(f"No bot called '{bot_id}'.")
        name = self.bots.get(bot_id)["name"] if bot_id else "Nuero AI"
        return self._make_image(self.scrubber.mask(prompt), bot_id, name, requested=True)

    def council(self, user_input: str, bot_ids=None, books=None):
        """Asks several bots at once (each with its own + shared memories) and yields each
        result as soon as it's ready: the reply dict, or {"bot", "bot_name", "error"}."""
        bots = [self.bots.get(b) for b in bot_ids] if bot_ids else list(self.bots.bots)
        bots = [b for b in bots if b]
        if not bots:
            raise ProviderError("No bots to ask. Add bots first, or pick at least one council member.")
        with ThreadPoolExecutor(max_workers=max(1, min(COUNCIL_PARALLEL, len(bots)))) as pool:
            futures = {pool.submit(self.respond, user_input, b["id"], True, None, books): b for b in bots}
            for future in as_completed(futures):
                bot = futures[future]
                try:
                    yield future.result()
                except Exception as e:
                    yield {"bot": bot["id"], "bot_name": bot["name"], "error": str(e)}
