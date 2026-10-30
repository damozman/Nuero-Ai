import datetime
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.bots import get_bots, SHARED
from core.images import get_studio, IMAGE_MARKER, AUTO_IMAGE_NOTE
from core.library import get_library, ARCHIVE_MARKER, ARCHIVE_NOTE
from core import music, profile
from core.music import SONG_MARKER, SONG_NOTE, SONG_REQUEST_NOTE, NO_SONG_TOOL_NOTE, MAX_SONGS
from core.providers import PROVIDERS, ProviderError
from core.scrubber import MemoryScrubber
from core.vector_engine import get_vault

SETTINGS_PATH = "config/ai_settings.json"  # chosen AI and the last model used with each
# How many council members are asked at once; lower it if the AI service rate-limits you
COUNCIL_PARALLEL = int(os.environ.get("NUERO_COUNCIL_PARALLEL", "12"))
# Recent messages of the same conversation sent with each reply. Cloud models read long conversations well;
# the local model has a small window and slows down, so it gets less. Older messages live on in a running
# summary (see summarize) - so long, deep threads keep their continuity.
HISTORY_LIMITS = {
    "cloud": {"turns": int(os.environ.get("NUERO_HISTORY_TURNS", "60")), "per_message": 8000, "total": 80000},
    "local": {"turns": int(os.environ.get("NUERO_LOCAL_HISTORY_TURNS", "12")), "per_message": 2500, "total": 14000},
}
SUMMARY_CHARS = 6000  # the running summary of older parts of a conversation
SONG_COOLDOWN = float(os.environ.get("NUERO_SONG_COOLDOWN_HOURS", "3")) * 3600  # unasked songs: once per window
ARCHIVE_COOLDOWN = float(os.environ.get("NUERO_ARCHIVE_COOLDOWN_HOURS", "1")) * 3600
COUNCIL_TURNS = 14  # recent lines of the council conversation every member sees
MIN_MEMORY_SCORE = float(os.environ.get("NUERO_MIN_MEMORY_SCORE", "0.2"))  # relevance cutoff (cosine similarity)
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
            "history_turns": self.history_limits()["turns"],  # recent messages sent raw; older ones get summarized
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
                return profile.with_system_name(f.read())
        return profile.with_system_name("You are {system_name}, the system helper for this household's AI app.")

    def chat(self, user_input: str, bot_id=None, history=None) -> str:
        return self.respond(user_input, bot_id, history=history)["reply"]

    # ---- who is who ----

    @property
    def user_name(self):
        return profile.user_name()

    def set_user_name(self, name):
        profile.save(user_name=name or "")

    def _node_identity(self, bot):
        """Stated plainly up front: small models otherwise mix up who they are and who they're talking to."""
        user = self.user_name or "the user"
        return (f"Your name is {bot['name']}. You are one of the Family Nodes: the AI personalities who live in "
                f"this household's {profile.system_name()} app. Always speak as yourself, {bot['name']}, in the first person "
                f"(\"I\", never \"{bot['name']} thinks\"). You are talking with {user}, a human - not another "
                f"Family Node.")

    @staticmethod
    def _summary_of(bot):
        """One short line about a Family Node, from the start of its personality, for the council roster."""
        text = re.sub(r"\s+", " ", bot.get("personality", "")).strip()
        first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0] if text else ""
        return (first[:160] + "…") if len(first) > 160 else first or "(no personality written yet)"

    def history_limits(self):
        return HISTORY_LIMITS["local" if self.provider_id == "ollama" else "cloud"]

    def _clean_history(self, history):
        """The most recent turns of this conversation that fit the current AI's limits, newest kept first,
        trimmed and masked like everything else that leaves the machine."""
        limits = self.history_limits()
        turns = [t for t in history or [] if t.get("role") in ("user", "assistant") and str(t.get("text", "")).strip()]
        kept, total = [], 0
        for t in reversed(turns[-limits["turns"]:]):
            text = str(t["text"])[:limits["per_message"]]
            if total + len(text) > limits["total"]:
                break
            kept.append({"role": t["role"], "text": self.scrubber.mask(text)})
            total += len(text)
        return list(reversed(kept))

    def summarize(self, messages, previous="", bot_id=None):
        """Folds older messages of a long conversation into its running summary, so a thread keeps its
        continuity after those messages fall out of the recent window. Returns the new summary."""
        bot = self.bots.get(bot_id) if bot_id else None
        name = bot["name"] if bot else profile.system_name()
        user = self.user_name or "the user"
        lines = []
        for t in messages or []:
            text = str(t.get("text", "")).strip()
            if text and t.get("role") in ("user", "assistant"):
                lines.append(f"{user if t['role'] == 'user' else name}: {text[:4000]}")
        if not lines:
            return previous
        system = (
            f"You keep the running memory of one long, ongoing conversation between {user} and {name}. "
            f"Update the summary so far with the new messages. Keep what matters for continuing the "
            f"conversation naturally: topics and how they developed, what {user} shared about their life, "
            f"feelings and moods, opinions, decisions, promises and plans, questions still open, running jokes "
            f"and anything {name} said it would remember. Write it in the third person as compact notes, at most "
            f"about 500 words. Drop small talk. Output only the updated summary."
        )
        prompt = (f"SUMMARY SO FAR:\n{self.scrubber.mask(previous) or '(none yet)'}\n\n"
                  f"NEW MESSAGES:\n" + self.scrubber.mask("\n".join(lines)))
        summary = self.provider.generate(self.model, prompt, system, None)
        return self.scrubber.unmask(summary.strip())[:SUMMARY_CHARS]

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
                council_context=None, memory_query=None, others=None, avoid_songs=None, summary=None) -> dict:
        """One reply. bot_id=None is the system helper (shared memories); a bot searches its own + shared.
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
            identity, name, partitions = self._get_identity_frequency(), profile.system_name(), [SHARED]
            if self.user_name:
                identity = f"{identity}\n\nYou are talking with {self.user_name}."
        active_skill = self.bots.skill(bot_id, skill) if bot_id and skill else None
        if active_skill:
            identity = (f"{identity}\n\n[ACTIVE SKILL: {active_skill['name']}]\n"
                        f"Apply this skill in your reply, staying in character:\n{active_skill['instructions']}")
        if council_context:
            identity = f"{identity}\n\n{council_context}"
        # Family Nodes may create images on their own, except in Council (one question could trigger a dozen)
        # and on the local AI, where the point is that nothing leaves the computer. The main system helper is the
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
        # "I'm open for all recommendations" counts as asking when the conversation is about music
        recent = council_context or " ".join(str(t.get("text", "")) for t in (history or [])[-4:])
        asked_for_music = music.wants_music(user_input, recent)
        may_share_music = bool(bot_id) and bot.get("music", True) and (
            asked_for_music or (not council and not library.recently_added(bot_id, "playlist", SONG_COOLDOWN)))
        if may_share_music:
            identity = f"{identity}\n\n{SONG_REQUEST_NOTE if asked_for_music else SONG_NOTE}"
        elif bot_id:
            identity = f"{identity}\n\n{NO_SONG_TOOL_NOTE}"

        # Only memories actually related to the message: the 3 closest are always returned, and loosely
        # related ones (e.g. old song shares on a privacy question) otherwise steer the reply off topic
        memories = [m for m in self.vault.search_memories(memory_query or user_input, limit=3, partitions=partitions)
                    if m.score >= MIN_MEMORY_SCORE]
        raw_context = "\n---\n".join([m.payload["text"] for m in memories])
        masked_context = self.scrubber.mask(raw_context)
        masked_prompt = self.scrubber.mask(user_input)
        identity_frequency = self.scrubber.mask(identity)

        system_instruction = f"""
        {identity_frequency}

        Things you remember (background for you - use them naturally, and never quote or mention this list):
        {masked_context or '(nothing relevant)'}
        """
        if summary and summary.strip():
            system_instruction += (
                "\nEarlier in this same conversation (a summary of what came before the recent messages - "
                "it's your shared history, so draw on it naturally):\n"
                + self.scrubber.mask(summary.strip()[:SUMMARY_CHARS]) + "\n"
            )
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
        reply = music.strip_placeholders(music.strip_markers(reply))
        songs = self._share_songs(song_marks, bot_id, name, avoid_songs) if song_marks and may_share_music else []

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

    def _share_songs(self, marks, bot_id, name, avoid=None):
        """avoid: songs others already shared this Council round (no echoing each other's picks)."""
        queries = [(self.scrubber.unmask(q), self.scrubber.unmask(why or "")) for q, why in marks]
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            cards = list(pool.map(lambda qw: music.lookup(qw[0]), queries))
        bot = self.bots.get(bot_id) if bot_id else None
        playlist = get_library().playlist_for(bot_id or "main", name, bot["color"] if bot else "#0096ff")
        shared = []
        for card, (_, why) in zip(cards, queries):
            if not card.get("found"):
                print(f"\n[MUSIC]: Skipped \"{card['title']}\" by {card['artist']} - not found in Apple or Deezer "
                      f"(probably not a real song)")
                continue
            if get_library().has_song(playlist["id"], card):
                continue  # never the same song twice in a playlist
            if any(music.same_song(card, s) for s in shared + list(avoid or [])):
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
        name = self.bots.get(bot_id)["name"] if bot_id else profile.system_name()
        return self._make_image(self.scrubber.mask(prompt), bot_id, name, requested=True)

    def _council_context(self, me, members, lines, newcomers=()):
        """What one member is told: who else is in the room, who just joined, and what has been said so far.
        Ends with a plain reminder of who is speaking: small models otherwise slip into another member's voice."""
        user = self.user_name or "the user"
        others = "\n".join(f"- {b['name']}: {self._summary_of(b)}" for b in members if b["id"] != me["id"])
        convo = "\n".join(lines[-COUNCIL_TURNS:]) or "(nothing yet - this is the start)"
        other_names = [b["name"] for b in members if b["id"] != me["id"]]
        if me["name"] in newcomers:
            joined = "You have just joined this council - this is your first time here.\n\n"
        elif newcomers:
            joined = (f"New here: {', '.join(newcomers)} just joined and hasn't spoken yet. You are not new - you "
                      f"have been part of this council.\n\n")
        else:
            joined = ""
        not_you = " or ".join(other_names + [user])
        return (
            f"[THE COUNCIL]\nYou are in a group conversation with {user} (a human) and other Family Nodes. "
            f"Everyone here sees everything that is said.\nThe other Family Nodes present:\n"
            f"{others or '- (no one else this time)'}\n\n"
            f"How to take part: speak only as yourself, {me['name']}, in the first person. If someone asked you a "
            f"question, answer it first - with real substance: an opinion, an example, something you know. Then "
            f"respond to what {user} and the others actually said. You may end by asking another Family Node one "
            f"question by name, but a reply that is only a question is not a real answer. Others will answer "
            f"when it is their turn, and {user} will answer for themself. Your reply is ONLY "
            f"your own next message: never write lines for {user} or the others (no \"{user}:\" or "
            f"\"Name:\" lines), never answer on their behalf, and don't narrate. Don't repeat or paraphrase what "
            f"someone else already said - add your own angle, true to your own personality. Keep it conversational and "
            f"fairly short.\n\n"
            f"{joined}"
            f"[THE CONVERSATION SO FAR]\n{convo}\n\n"
            f"[YOUR TURN]\nYou are {me['name']} - not {not_you}. When asked about yourself, answer about "
            f"{me['name']} from your own personality, even if someone else answered the same question. Write only "
            f"{me['name']}'s next message."
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
                      f"another Family Node by name. If they asked you something, answer it properly first - share "
                      f"your actual view or an example. Then react. Bring something new: a concrete example, a story, "
                      f"a different opinion or a respectful disagreement. Don't open with praise like 'your insights "
                      f"are fascinating', and don't end with a question every time - only now and then. "
                      f"Don't address {user} this time.)")
            memory_query = " ".join(lines[-3:]) or "council conversation"
        present = list(members)
        # Members who haven't spoken yet in a conversation that is already under way have just joined
        spoken = {line.split(":", 1)[0] for line in lines}
        newcomers = [b["name"] for b in present if b["name"] not in spoken] if spoken - {user} else []
        if not keep_going:
            prompt, memory_query = user_input, None
            lines.append(f"{user}: {user_input}")
            # Speaking to someone by name means they answer - and only they, unless everyone is addressed
            lowered = user_input.lower()
            named = [b for b in members if re.search(rf"\b{re.escape(b['name'].lower())}\b", lowered)]
            everyone = re.search(r"\b(everyone|everybody|all of you|you all|y'all|you guys|both of you|each of "
                                 r"you|the council|you two|guys)\b", lowered)
            if named and not everyone:
                members = sorted(named, key=lambda b: lowered.find(b["name"].lower()))
            else:
                members.sort(key=lambda b: lowered.find(b["name"].lower()) if b in named else 10**6)

        shared_songs = []  # songs shared so far this round, so members don't echo each other's picks

        def ask(member, convo):
            context = self.scrubber.mask(self._council_context(member, present, convo, newcomers))
            return self.respond(prompt, member["id"], True, None, books, council_context=context,
                                memory_query=memory_query,
                                others=[b["name"] for b in present if b["id"] != member["id"]],
                                avoid_songs=list(shared_songs))

        if together:
            for member in members:
                try:
                    result = ask(member, lines)
                    shared_songs.extend(result.get("songs", []))
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
