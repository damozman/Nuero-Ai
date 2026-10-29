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
from core.pipeline import get_pipeline
from core.providers import ProviderError
from core.vector_engine import get_vault

app = FastAPI()
os.makedirs(IMAGES_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


class TextIn(BaseModel):
    text: str


class BotTextIn(BaseModel):
    text: str
    bot: str | None = None  # None = main Nuero AI / shared memory


class ChatIn(BotTextIn):
    history: list[dict] | None = None  # earlier turns of this conversation: [{"role", "text"}]


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
        raise ValueError(f"No bot called '{bot_id}'.")
    return bot_id or SHARED


# ---------------------------------------------------------------- AI service

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
        return get_pipeline().respond(body.text, body.bot or None, history=body.history)
    except Exception as e:
        return _error(str(e), 500)


class CouncilIn(BaseModel):
    text: str
    bots: list[str] | None = None  # None = every bot


@app.post("/api/council")
def post_council(body: CouncilIn):
    """Streams one JSON line per bot as each answers, so fast bots show up without waiting for slow ones."""
    if not body.text.strip():
        return _error("Message is empty.")
    pipeline = get_pipeline()
    if not (body.bots or pipeline.bots.bots):
        return _error("No bots yet. Add some in the Bots panel first.")

    def lines():
        try:
            for result in pipeline.council(body.text, body.bots):
                yield json.dumps(result) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


# ---------------------------------------------------------------- images

@app.post("/api/image")
def post_image(body: BotTextIn):
    """An image the user asked for, in the conversation with body.bot (None = Nuero AI)."""
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
    return {"bots": [dict(b, images=b.get("images", True), folders=daemon.folders_of(b["id"]) if daemon else [])
                     for b in bots]}


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
            raise ValueError("The built-in vault folder is shared memory; pick a different folder for a bot.")
        if folder and daemon.owner_of_folder(folder) != SHARED and folder in daemon.extra_folders:
            raise ValueError("That folder already belongs to another bot.")
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


@app.post("/api/bots/update")
def update_bot(body: BotUpdateIn):
    try:
        bot = get_bots().update(body.id, body.name, body.personality, body.images)
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
    except ValueError as e:
        return _error(str(e))
    return _bot_list(daemon)


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
        return f.read()


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        page = f.read()
    # Help panel shows real locations on this machine
    return (page
            .replace("{{VAULT_PATH}}", html.escape(os.path.abspath("storage/vault")))
            .replace("{{CONFIG_PATH}}", html.escape(os.path.abspath("config")))
            .replace("{{IMAGES_PATH}}", html.escape(os.path.abspath(IMAGES_DIR))))


def run_dashboard():
    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="warning")
