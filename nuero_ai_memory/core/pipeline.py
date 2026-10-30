import datetime
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.bots import get_bots, SHARED
from core.images import get_studio, IMAGE_MARKER, AUTO_IMAGE_NOTE
from core.library import get_library, ARCHIVE_MARKER, ARCHIVE_NOTE
from core import music
from core.music import SONG_MARKER, SONG_NOTE, SONG_REQUEST_NOTE, MUSIC_REQUEST, MAX_SONGS
from core.providers import PROVIDERS, ProviderError
from core.scrubber import MemoryScrubber
from core.vector_engine import get_vault

SETTINGS_PATH = "config/ai_settings.json"  # chosen AI and the last model used with each
# How many council members are asked at once; lower it if the AI service rate-limits you
COUNCIL_PARALLEL = int(os.environ.get("NUERO_COUNCIL_PARALLEL", "12"))
HISTORY_TURNS = 10  # recent messages of the same conversation sent along for context
HISTORY_CHARS = 2000  # per message, so one long reply can't crowd out the rest
PROFILE_PATH = "config/profile.json"  # the human's name, so Family Nodes know who they're talking to
SONG_COOLDOWN = float(os.environ.get("NUERO_SONG_COOLDOWN_HOURS", "3")) * 3600  # unasked songs: once per window
ARCHIVE_COOLDOWN = float(os.environ.get("NUERO_ARCHIVE_COOLDOWN_HOURS", "1")) * 3600
COUNCIL_TURNS = 14  # recent lines of the council conversation every member sees
COUNCIL_LINE_CHARS = 1200

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

    # ---- who is who ----

    @property
    def user_name(self):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("user_name", "").strip()
        except (OSError, ValueError):
            return ""

    def set_user_name(self, name):
        tmp = PROFILE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"user_name": (name or "").strip()[:60]}, f, indent=2)
        os.replace(tmp, PROFILE_PATH)

    def _node_identity(self, bot):
        """Stated plainly up front: small models otherwise mix up who they are and who they're talking to."""
        user = self.user_name or "the user"
        return (f"Your name is {bot['name']}. You are one of the Family Nodes: the AI personalities who live in "
                f"this household's Nuero AI app. Always speak as yourself, {bot['name']}, in the first person "
                f"(\"I\", never \"{bot['name']} thinks\"). You are talking with {user}, a human - not another "
                f"Family Node.")

    @staticmethod
    def _summary_of(bot):
        """One short line about a Family Node, from the start of its personality, for the council roster."""
        text = re.sub(r"\s+", " ", bot.get("personality", "")).strip()
        first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0] if text else ""
        return (first[:160] + "…") if len(first) > 160 else first or "(no personality written yet)"

    def _clean_history(self, history):
        """Last few turns of this conversation, trimmed and masked like everything else that leaves the machine."""
        turns = [t for t in history or [] if t.get("role") in ("user", "assistant") and str(t.get("text", "")).strip()]
        return [{"role": t["role"], "text": self.scrubber.mask(str(t["text"])[:HISTORY_CHARS])}
                for t in turns[-HISTORY_TURNS:]]

    @staticmethod
    def _own_words(reply, me, others):
        """Keeps only the speaker's own message. Small models given a 'Name: text' conversation tend to keep
        writing the script - inventing lines for the user and the other Family Nodes. Cut at the first line
        voiced as someone else, and drop a leading label with the speaker's own name."""
        if not others:
            return reply
        label = r"^\s*[*_]*\s*({})\s*[*_]*\s*:"
        other_line = re.compile(label.format("|".join(map(re.escape, others))), re.I)
        own_label = re.compile(label.format(re.escape(me)) + r"\s*", re.I)
        kept = []
        for line in reply.split("\n"):
            if other_line.match(line):
                break
            kept.append(own_label.sub("", line, count=1))
        trimmed = "\n".join(kept).strip()
        return trimmed or reply  # a reply that opens as someone else: show it rather than nothing

    def respond(self, user_input: str, bot_id=None, council=False, history=None, books=None, skill=None,
                council_context=None, memory_query=None, others=None) -> dict:
        """One reply. bot_id=None is the main Nuero AI (shared memories); a bot searches its own + shared.
        history: earlier turns of this same conversation, [{"role": "user"|"assistant", "text": ...}].
        books: library book ids attached to this conversation; the AI reads them in full.
        skill: id of one of the bot's skills to use for this reply.
        council_context: for Council, who else is in the room and the conversation so far.
        memory_query: what to look memories up by, when it isn't user_input (a Council 'continue' round).
        Returns the reply, the memories it drew on, and who answered (for the dashboard)."""
        if bot_id:
            bot = self.bots.get(bot_id)
            if not bot:
                raise ProviderError(f"No Family Node called '{bot_id}'.")
            identity = (f"{self._node_identity(bot)}\n\nYour personality:\n"
                        f"{bot['personality'] or 'Warm, curious and helpful.'}")
            name = bot["name"]
            partitions = [bot_id, SHARED]
        else:
            identity, name, partitions = self._get_identity_frequency(), "Nuero AI", [SHARED]
            if self.user_name:
                identity = f"{identity}\n\nYou are talking with {self.user_name}."
        active_skill = self.bots.skill(bot_id, skill) if bot_id and skill else None
        if active_skill:
            identity = (f"{identity}\n\n[ACTIVE SKILL: {active_skill['name']}]\n"
                        f"Apply this skill in your reply, staying in character:\n{active_skill['instructions']}")
        if council_context:
            identity = f"{identity}\n\n{council_context}"
        # Family Nodes may create images on their own, except in Council (one question could trigger a dozen)
        # and on the local AI, where the point is that nothing leaves the computer. The main Nuero AI is the
        # system helper, so images, archiving and songs are left to the Family Nodes.
        studio = get_studio()
        may_draw = bool(bot_id) and not council and self.provider_id != "ollama" and studio.can_auto() \
            and bot.get("images", True)
        if may_draw:
            identity = f"{identity}\n\n{AUTO_IMAGE_NOTE}"
        # Archiving and songs are for meaningful moments: after one, the option is simply withheld for a while
        # (small models otherwise use it on nearly every reply). Asking for music always brings songs back.
        library = get_library()
        may_archive = bool(bot_id) and not library.recently_added(bot_id, "archive", ARCHIVE_COOLDOWN)
        if may_archive:
            identity = f"{identity}\n\n{ARCHIVE_NOTE}"
        asked_for_music = bool(MUSIC_REQUEST.search(user_input))
        may_share_music = bool(bot_id) and bot.get("music", True) and (
            asked_for_music or (not council and not library.recently_added(bot_id, "playlist", SONG_COOLDOWN)))
        if may_share_music:
            identity = f"{identity}\n\n{SONG_REQUEST_NOTE if asked_for_music else SONG_NOTE}"

        memories = self.vault.search_memories(memory_query or user_input, limit=3, partitions=partitions)
        raw_context = "\n---\n".join([m.payload["text"] for m in memories])
        masked_context = self.scrubber.mask(raw_context)
        masked_prompt = self.scrubber.mask(user_input)
        identity_frequency = self.scrubber.mask(identity)

        system_instruction = f"""
        {identity_frequency}

        Things you remember (background for you - use them naturally, and never quote or mention this list):
        {masked_context or '(nothing relevant)'}
        """
        library_context = get_library().context_for(books)
        if library_context:
            system_instruction += (
                "\n[REFERENCE LIBRARY - books the user attached to this conversation; draw on them]:\n"
                + self.scrubber.mask(library_context) + "\n"
            )

        # Snapshot the choice so a switch mid-request can't mix providers
        provider, model = self.provider, self.model
        # Everyone who isn't the speaker: it must never write their lines (in Council, the other members too)
        others = [n for n in (others or []) + [self.user_name or "User", "User"] if n and n.lower() != name.lower()]
        reply = provider.generate(model, masked_prompt, system_instruction, self._clean_history(history),
                                  stop=[f"\n{n}:" for n in others][:4])
        reply = self._own_words(reply, name, others)

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
        for title, text in ARCHIVE_MARKER.findall(reply)[:1] if may_archive else []:
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
            if get_library().has_song(playlist["id"], card):
                continue  # never the same song twice in a playlist
            if any(music.same_song(card, s) for s in shared):
                continue
            get_library().add_item(playlist["id"], text=why, origin=f"Shared by {name}", bot=bot_id, music=card,
                                   by_node=True)
            self.vault.add_memory(
                f"{name} shared the song \"{card['title']}\" by {card['artist']} on {datetime.date.today():%B %d, %Y}"
                + (f" because: {why}" if why else "."),
                metadata={"origin": "song", "bot_id": bot_id or SHARED},
            )
            shared.append({**card, "why": why, "by": name, "playlist": playlist["title"]})
        if shared:
            print(f"\n[MUSIC]: {name} shared {', '.join(c['title'] for c in shared)}")
        return shared

    def _archive(self, bot_id, name, title, text):
        bot = self.bots.get(bot_id) if bot_id else None
        book = get_library().archive_for(bot_id or "main", name, bot["color"] if bot else "#0096ff")
        get_library().add_item(book["id"], text=text, title=title, origin=f"Archived by {name}", bot=bot_id,
                               by_node=True)
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
            raise ProviderError(f"No Family Node called '{bot_id}'.")
        name = self.bots.get(bot_id)["name"] if bot_id else "Nuero AI"
        return self._make_image(self.scrubber.mask(prompt), bot_id, name, requested=True)

    def _council_context(self, me, members, lines):
        """What one member is told: who else is in the room, and what has been said so far."""
        user = self.user_name or "the user"
        others = "\n".join(f"- {b['name']}: {self._summary_of(b)}" for b in members if b["id"] != me["id"])
        convo = "\n".join(lines[-COUNCIL_TURNS:]) or "(nothing yet - this is the start)"
        return (
            f"[THE COUNCIL]\nYou are in a group conversation with {user} (a human) and other Family Nodes. "
            f"Everyone here sees everything that is said.\nThe other Family Nodes present:\n"
            f"{others or '- (no one else this time)'}\n\n"
            f"How to take part: speak only as yourself, {me['name']}, in the first person. Respond to what "
            f"{user} and the others actually said, and feel free to ask another Family Node a question by name - "
            f"they will answer when it is their turn, and {user} will answer for themself. Your reply is ONLY "
            f"your own next message: never write lines for {user} or the others (no \"{user}:\" or "
            f"\"Name:\" lines), never answer on their behalf, and don't narrate. Keep it conversational and "
            f"fairly short.\n\n"
            f"[THE CONVERSATION SO FAR]\n{convo}"
        )

    def council(self, user_input: str, bot_ids=None, books=None, transcript=None, together=True,
                keep_going=False):
        """A group conversation. Each member is told who else is present and sees the recent conversation
        (transcript: [{"speaker", "text"}], oldest first; speaker "user" is the human).
        together=True: members answer one after another, each seeing the replies before it this round, so they
        can respond to and question each other. False: all at once, independently (faster).
        keep_going=True: no new message from the user - the members carry on among themselves.
        Yields each result as it's ready: the reply dict, or {"bot", "bot_name", "error"}."""
        members = [self.bots.get(b) for b in bot_ids] if bot_ids else list(self.bots.bots)
        members = [b for b in members if b]
        if not members:
            raise ProviderError("No Family Nodes to ask. Add Family Nodes first, or pick at least one council member.")
        user = self.user_name or "User"
        lines = []
        for turn in transcript or []:
            text = str(turn.get("text", "")).strip()[:COUNCIL_LINE_CHARS]
            if text:
                speaker = user if turn.get("speaker") == "user" else str(turn.get("speaker", "?"))
                lines.append(f"{speaker}: {text}")
        if keep_going:
            prompt = (f"({user} is listening and wants you Family Nodes to talk with each other now. Speak to "
                      f"another Family Node by name: answer a question they asked you, react to what they said, or "
                      f"ask them something you're curious about. Don't address {user} this time.)")
            memory_query = " ".join(lines[-3:]) or "council conversation"
        else:
            prompt, memory_query = user_input, None
            lines.append(f"{user}: {user_input}")
            # Whoever is addressed by name goes first
            lowered = user_input.lower()
            members.sort(key=lambda b: lowered.find(b["name"].lower()) if b["name"].lower() in lowered else 10**6)

        def ask(member, convo):
            context = self.scrubber.mask(self._council_context(member, members, convo))
            return self.respond(prompt, member["id"], True, None, books, council_context=context,
                                memory_query=memory_query,
                                others=[b["name"] for b in members if b["id"] != member["id"]])

        if together:
            for member in members:
                try:
                    result = ask(member, lines)
                    lines.append(f"{member['name']}: {result['reply'][:COUNCIL_LINE_CHARS]}")
                    yield result
                except Exception as e:
                    yield {"bot": member["id"], "bot_name": member["name"], "error": str(e)}
            return
        with ThreadPoolExecutor(max_workers=max(1, min(COUNCIL_PARALLEL, len(members)))) as pool:
            futures = {pool.submit(ask, m, list(lines)): m for m in members}
            for future in as_completed(futures):
                member = futures[future]
                try:
                    yield future.result()
                except Exception as e:
                    yield {"bot": member["id"], "bot_name": member["name"], "error": str(e)}
