import html
import json
import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from core.bots import get_bots, find_persona_text, SHARED
from core.images import get_studio, IMAGES_DIR
from core.library import get_library
from core import avatars, profile
from core.pipeline import get_pipeline
from core.providers import ProviderError
from core.vector_engine import get_vault

app = FastAPI()
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(avatars.AVATARS_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/avatars", StaticFiles(directory=avatars.AVATARS_DIR), name="avatars")


class TextIn(BaseModel):
    text: str


class BotTextIn(BaseModel):
    text: str
    bot: str | None = None  # None = the system helper / shared memory


class ChatIn(BotTextIn):
    history: list[dict] | None = None  # earlier turns of this conversation: [{"role", "text"}]
    books: list[str] | None = None  # library books attached to this conversation
    skill: str | None = None  # one of the bot's skills, for this reply


class AiChoiceIn(BaseModel):
    provider: str
    model: str | None = None


def _error(message, status=400):
    return JSONResponse({"error": message}, status_code=status)


def _daemon_or_error():
    from daemons.watcher import get_daemon
    daemon = get_daemon()
    if daemon is None:
        return None, _error("The file watcher isn't running. Start the app with main.py.", 503)
    return daemon, None


def _check_bot(bot_id):
    if bot_id and bot_id != SHARED and not get_bots().get(bot_id):
        raise ValueError(f"No Family Node called '{bot_id}'.")
    return bot_id or SHARED


# ---------------------------------------------------------------- AI service

class ProfileIn(BaseModel):
    user_name: str | None = None
    system_name: str | None = None  # what the whole system is called (header, helper, guide)


def _profile():
    return {"user_name": profile.user_name(), "system_name": profile.system_name()}


@app.get("/api/profile")
def get_profile():
    return _profile()


@app.post("/api/profile")
def post_profile(body: ProfileIn):
    renamed = body.system_name is not None and body.system_name.strip() != profile.system_name()
    profile.save(user_name=body.user_name, system_name=body.system_name)
    if renamed:
        get_library().refresh_guide()
    return _profile()


@app.get("/api/ai")
def get_ai():
    return get_pipeline().ai_status()


@app.post("/api/ai")
def post_ai(body: AiChoiceIn):
    try:
        get_pipeline().set_ai(body.provider, body.model)
    except ProviderError as e:
        return _error(str(e))
    return get_pipeline().ai_status()


# ---------------------------------------------------------------- chat & council

@app.post("/api/chat")
def post_chat(body: ChatIn):
    if not body.text.strip():
        return _error("Message is empty.")
    try:
        return get_pipeline().respond(body.text, body.bot or None, history=body.history, books=body.books,
                                      skill=body.skill)
    except Exception as e:
        return _error(str(e), 500)


class CouncilIn(BaseModel):
    text: str = ""
    bots: list[str] | None = None  # None = every Family Node
    books: list[str] | None = None
    transcript: list[dict] | None = None  # recent council conversation: [{"speaker", "text"}]
    together: bool = True  # one after another, seeing each other's replies (False: all at once)
    keep_going: bool = False  # no new message: the Family Nodes carry on among themselves


@app.post("/api/council")
def post_council(body: CouncilIn):
    """Streams one JSON line per Family Node as each answers, so replies appear as they come."""
    if not body.text.strip() and not body.keep_going:
        return _error("Message is empty.")
    pipeline = get_pipeline()
    if not (body.bots or pipeline.bots.bots):
        return _error("No Family Nodes yet. Add some in the Family Nodes panel first.")

    def lines():
        try:
            for result in pipeline.council(body.text, body.bots, body.books, body.transcript,
                                           body.together, body.keep_going):
                yield json.dumps(result) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


# ---------------------------------------------------------------- library

@app.get("/api/library")
def get_library_shelf():
    return {"books": get_library().summaries()}


@app.get("/api/library/book/{book_id}")
def get_book(book_id: str):
    book = get_library().get(book_id)
    return book if book else _error("That book doesn't exist any more.", 404)


class BookIn(BaseModel):
    id: str | None = None
    title: str | None = None
    emoji: str | None = None
    color: str | None = None


@app.post("/api/library/books")
def create_book(body: BookIn):
    try:
        return get_library().create_book(body.title, body.emoji, body.color)
    except ValueError as e:
        return _error(str(e))


@app.post("/api/library/books/update")
def update_book(body: BookIn):
    try:
        return get_library().update_book(body.id, body.title, body.emoji, body.color)
    except ValueError as e:
        return _error(str(e))


@app.post("/api/library/books/seen")
def mark_book_seen(body: TextIn):
    """She opened the book: clear the 'new' flags on pages Family Nodes added."""
    try:
        get_library().mark_seen(body.text)
    except ValueError as e:
        return _error(str(e))
    return {"books": get_library().summaries()}


@app.post("/api/library/books/delete")
def delete_book(body: TextIn):
    try:
        get_library().delete_book(body.text)
    except ValueError as e:
        return _error(str(e))
    return {"books": get_library().summaries()}


class ItemIn(BaseModel):
    book: str
    text: str | None = None
    title: str | None = None
    source: str | None = None
    origin: str | None = None
    bot: str | None = None
    image: str | None = None  # file name in storage/images
    music: dict | None = None  # a song card shared by a bot
    path: str | None = None  # add a whole text file instead of text


@app.post("/api/library/items")
def add_book_item(body: ItemIn):
    try:
        if body.path:
            return get_library().add_file(body.book, body.path)
        image = os.path.basename(body.image) if body.image else None  # only files from the images folder
        if image and not os.path.isfile(os.path.join(IMAGES_DIR, image)):
            raise ValueError("That image file no longer exists.")
        return get_library().add_item(body.book, body.text, body.title, body.source or "",
                                      body.origin or "", body.bot, image, body.music)
    except ValueError as e:
        return _error(str(e))


class ItemRemoveIn(BaseModel):
    book: str
    item: str


@app.post("/api/library/items/delete")
def remove_book_item(body: ItemRemoveIn):
    try:
        get_library().remove_item(body.book, body.item)
    except ValueError as e:
        return _error(str(e))
    return get_library().get(body.book)


@app.get("/api/memories/search")
def search_all_memories(q: str = "", limit: int = 20):
    """Search every memory (shared and every bot's) so the user can pick pieces for a book."""
    if not q.strip():
        return {"results": []}
    results = get_vault().search_memories(q, limit=max(1, min(limit, 50)))
    return {"results": [
        {
            "id": r.id,
            "text": r.payload.get("text", ""),
            "source": str(r.payload.get("metadata", {}).get("source", "")).split(":", 1)[-1],
            "bot": r.payload.get("metadata", {}).get("bot_id", SHARED),
            "score": round(r.score, 3),
        }
        for r in results
    ]}


# ---------------------------------------------------------------- images

@app.post("/api/image")
def post_image(body: BotTextIn):
    """An image the user asked for, in the conversation with body.bot (None = the system helper)."""
    if not body.text.strip():
        return _error("Describe the image you want.")
    try:
        return get_pipeline().create_image(body.text, body.bot or None)
    except Exception as e:
        return _error(str(e), 500)


@app.get("/api/images/settings")
def get_image_settings():
    return get_studio().status()


class ImageSettingsIn(BaseModel):
    provider: str | None = None
    daily_cap: int | None = None
    main_auto: bool | None = None


@app.post("/api/images/settings")
def post_image_settings(body: ImageSettingsIn):
    try:
        return get_studio().update(body.provider, body.daily_cap, body.main_auto)
    except ValueError as e:
        return _error(str(e))


# ---------------------------------------------------------------- memories & graph

@app.get("/api/network")
def get_network():
    return get_vault().fetch_all_nodes_and_edges()


@app.post("/api/memory")
def post_memory(body: BotTextIn):
    if not body.text.strip():
        return _error("Memory is empty.")
    try:
        owner = _check_bot(body.bot)
    except ValueError as e:
        return _error(str(e))
    get_vault().add_memory(body.text, metadata={"origin": "dashboard", "bot_id": owner})
    return {"ok": True}


@app.post("/api/memory/forget")
def forget_memory(body: TextIn):
    get_vault().remove_point(body.text)
    return {"ok": True}


# ---------------------------------------------------------------- bots

def _bot_list(daemon=None):
    bots = get_bots().bots
    urls = avatars.all_urls()
    return {
        "bots": [dict(b, images=b.get("images", True), music=b.get("music", True), skills=b.get("skills", []), avatar=urls.get(b["id"]),
                      folders=daemon.folders_of(b["id"]) if daemon else [])
                 for b in bots],
        "main": {"avatar": urls.get("main")},
        "council": {"avatar": urls.get("council")},
    }


@app.get("/api/bots")
def get_bot_list():
    from daemons.watcher import get_daemon
    return _bot_list(get_daemon())


class BotCreateIn(BaseModel):
    name: str | None = None
    folder: str | None = None
    personality: str | None = None


@app.post("/api/bots")
def create_bot(body: BotCreateIn):
    """New bot, optionally fed by a folder (its name and personality are pre-filled from the folder)."""
    daemon, err = _daemon_or_error()
    if err:
        return err
    try:
        folder = daemon.normalize_folder(body.folder) if body.folder and body.folder.strip() else None
        if folder and folder == daemon.vault_dir:
            raise ValueError("The built-in vault folder is shared memory; pick a different folder for a Family Node.")
        if folder and daemon.owner_of_folder(folder) != SHARED and folder in daemon.extra_folders:
            raise ValueError("That folder already belongs to another Family Node.")
        name = (body.name or "").strip() or (os.path.basename(folder) if folder else "")
        personality = body.personality if body.personality and body.personality.strip() else (
            find_persona_text(folder) if folder else "")
        bot = get_bots().add(name, personality)
        if folder:
            if folder in daemon.extra_folders:
                daemon.set_folder_owner(folder, bot["id"])  # was shared; now this bot's
            else:
                daemon.add_folder(folder, bot["id"])
    except ValueError as e:
        return _error(str(e))
    return {"bot": bot, **_bot_list(daemon)}


class BotUpdateIn(BaseModel):
    id: str
    name: str | None = None
    personality: str | None = None
    images: bool | None = None
    music: bool | None = None


@app.post("/api/bots/update")
def update_bot(body: BotUpdateIn):
    try:
        bot = get_bots().update(body.id, body.name, body.personality, body.images, body.music)
    except ValueError as e:
        return _error(str(e))
    return {"bot": bot}


@app.post("/api/bots/delete")
def delete_bot(body: TextIn):
    daemon, err = _daemon_or_error()
    if err:
        return err
    try:
        get_bots().require(body.text)
        daemon.remove_bot(body.text)
        get_bots().delete(body.text)
        avatars.remove(body.text)
    except ValueError as e:
        return _error(str(e))
    return _bot_list(daemon)


class SkillIn(BaseModel):
    bot: str
    id: str | None = None
    name: str | None = None
    instructions: str | None = None


@app.post("/api/bots/skills")
def add_skill(body: SkillIn):
    try:
        return get_bots().add_skill(body.bot, body.name, body.instructions)
    except ValueError as e:
        return _error(str(e))


@app.post("/api/bots/skills/update")
def update_skill(body: SkillIn):
    try:
        return get_bots().update_skill(body.bot, body.id, body.name, body.instructions)
    except ValueError as e:
        return _error(str(e))


@app.post("/api/bots/skills/delete")
def delete_skill(body: SkillIn):
    try:
        get_bots().delete_skill(body.bot, body.id)
    except ValueError as e:
        return _error(str(e))
    return {"ok": True}


# ---------------------------------------------------------------- avatars

class AvatarIn(BaseModel):
    id: str  # a Family Node id, "main" for the system helper, or "council"
    data: str | None = None  # uploaded picture as a data: URL
    prompt: str | None = None  # for a generated one


def _avatar_owner(owner_id):
    if owner_id not in ("main", "council"):
        get_bots().require(owner_id)
    return owner_id


@app.post("/api/avatar")
def upload_avatar(body: AvatarIn):
    try:
        return {"avatar": avatars.save_data_url(_avatar_owner(body.id), body.data)}
    except ValueError as e:
        return _error(str(e))


@app.post("/api/avatar/generate")
def generate_avatar(body: AvatarIn):
    """Paints a portrait with the image AI, from the bot's personality unless a prompt is given."""
    try:
        owner = _avatar_owner(body.id)
        bot = get_bots().get(owner)
        if owner == "council":
            default = ("An emblem for 'the Council': a circle of distinct AI companions gathered together as one "
                       "family. Symbolic, warm, centered, simple background, square composition.")
        else:
            name = bot["name"] if bot else profile.system_name()
            about = (bot["personality"] if bot else "a friendly, futuristic personal AI assistant")[:400]
            default = (f"Portrait avatar for a character named {name}. Character: {about}. "
                       "Head and shoulders, centered, expressive, soft lighting, simple background, square composition.")
        prompt = (body.prompt or "").strip() or default
        pipeline = get_pipeline()
        info = get_studio().create(pipeline.scrubber.mask(prompt), auto=False)
        return {"avatar": avatars.save_file(owner, os.path.join(IMAGES_DIR, info["file"]))}
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        return _error(str(e), 500)


@app.post("/api/avatar/delete")
def delete_avatar(body: AvatarIn):
    avatars.remove(body.id)
    return {"ok": True}


# ---------------------------------------------------------------- watched folders

@app.get("/api/folders")
def get_folders():
    daemon, err = _daemon_or_error()
    if err:
        return err
    return {
        "folders": [{"path": f, "default": f == daemon.vault_dir, "exists": os.path.isdir(f),
                     "bot": daemon.owner_of_folder(f)}
                    for f in daemon.folders()]
    }


@app.post("/api/folders")
def add_folder(body: BotTextIn):
    daemon, err = _daemon_or_error()
    if err:
        return err
    try:
        return {"added": daemon.add_folder(body.text, _check_bot(body.bot))}
    except ValueError as e:
        return _error(str(e))


class FolderRemoveIn(BaseModel):
    text: str
    forget: bool = True


@app.post("/api/folders/remove")
def remove_folder(body: FolderRemoveIn):
    daemon, err = _daemon_or_error()
    if err:
        return err
    try:
        return {"files": daemon.remove_folder(body.text, forget=body.forget)}
    except ValueError as e:
        return _error(str(e))


@app.get("/graph", response_class=HTMLResponse)
def get_graph():
    """The original 3D memory graph, kept as its own page for later experiments."""
    with open("templates/graph.html", "r", encoding="utf-8") as f:
        return _with_names(f.read())


def _with_names(page):
    """Fills the system's current name into a page: as text, and as a JavaScript string."""
    name = profile.system_name()
    return (page.replace("{{SYSTEM_NAME_JS}}", json.dumps(name).replace("<", "\\u003c"))
                .replace("{{SYSTEM_NAME}}", html.escape(name)))


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        page = f.read()
    # Help panel shows real locations on this machine
    return (_with_names(page)
            .replace("{{VAULT_PATH}}", html.escape(os.path.abspath("storage/vault")))
            .replace("{{CONFIG_PATH}}", html.escape(os.path.abspath("config")))
            .replace("{{IMAGES_PATH}}", html.escape(os.path.abspath(IMAGES_DIR))))


def run_dashboard():
    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="warning")
