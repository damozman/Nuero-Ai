"""Image creation through Gemini or Grok (Claude can't make images), with a daily cap on
images bots create on their own. Files are kept in storage/images and served at /images/<file>."""
import base64
import datetime
import json
import os
import re
import threading
import time

from core.providers import PROVIDERS, ProviderError

IMAGES_DIR = "storage/images"
SETTINGS_PATH = "config/image_settings.json"
DEFAULT_DAILY_CAP = 20

# Bots add this line to a reply when they decide an image would help
IMAGE_MARKER = re.compile(r"\[\[\s*IMAGE\s*:\s*(.+?)\s*\]\]", re.I | re.S)
AUTO_IMAGE_NOTE = (
    "You can create an image. Only if a picture would genuinely help or fit the moment, end your reply with "
    "one line in exactly this form: [[IMAGE: a detailed visual description of the picture]]. "
    "At most one per reply. Never mention or explain this syntax."
)


class GeminiImager:
    id = "gemini"
    label = "Gemini"

    def __init__(self):
        self._model = os.environ.get("GEMINI_IMAGE_MODEL")

    def available(self):
        return PROVIDERS["gemini"].has_key()

    def _pick_model(self, client):
        """Newest Imagen model this key can use, else a Gemini model with image output."""
        names = [m.name.removeprefix("models/") for m in client.models.list()]
        imagen = sorted((n for n in names if n.startswith("imagen")), reverse=True)
        gemini_image = sorted((n for n in names if n.startswith("gemini") and "image" in n and "preview" not in n),
                              reverse=True)
        picked = (imagen or gemini_image or [None])[0]
        if not picked:
            raise ProviderError("This Gemini key has no image models available.")
        return picked

    def create(self, prompt):
        from google.genai import errors
        client = PROVIDERS["gemini"].client()
        try:
            model = self._model or self._pick_model(client)
            self._model = model
            if model.startswith("imagen"):
                response = client.models.generate_images(model=model, prompt=prompt, config={"number_of_images": 1})
                images = response.generated_images or []
                if not images or not images[0].image or not images[0].image.image_bytes:
                    reason = images[0].rai_filtered_reason if images else None
                    raise ProviderError("Gemini didn't return an image" + (f" ({reason})." if reason else ". Try rewording the prompt."))
                return images[0].image.image_bytes, images[0].image.mime_type or "image/png", model
            response = client.models.generate_content(
                model=model, contents=prompt, config={"response_modalities": ["IMAGE", "TEXT"]}
            )
            for part in (response.candidates[0].content.parts if response.candidates else []):
                if part.inline_data and part.inline_data.data:
                    data = part.inline_data.data
                    if isinstance(data, str):
                        data = base64.b64decode(data)
                    return data, part.inline_data.mime_type or "image/png", model
            raise ProviderError("Gemini didn't return an image. Try rewording the prompt.")
        except errors.APIError as e:
            if e.code in (400, 401, 403) and "key" in str(e.message).lower():
                raise ProviderError("Gemini rejected the API key. Check GEMINI_API_KEY.") from e
            if e.code == 429 or (e.code or 0) >= 500:
                raise ProviderError("Gemini's image service is busy right now. Try again in a minute.") from e
            raise ProviderError(f"Gemini couldn't create the image: {e.message}") from e


class GrokImager:
    id = "grok"
    label = "Grok"

    def __init__(self):
        self._model = os.environ.get("GROK_IMAGE_MODEL", "grok-imagine-image")

    def available(self):
        return PROVIDERS["grok"].has_key()

    def create(self, prompt):
        import grpc
        client = PROVIDERS["grok"].client()
        try:
            response = client.image.sample(prompt=prompt, model=self._model, image_format="base64")
            return response.image, "image/jpeg", self._model
        except grpc.RpcError as e:
            code, details = e.code(), str(e.details() or "")
            if code in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED) or "api key" in details.lower():
                raise ProviderError("Grok rejected the API key. Check XAI_API_KEY.") from e
            if code in (grpc.StatusCode.RESOURCE_EXHAUSTED, grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                raise ProviderError("Grok's image service is busy right now. Try again in a minute.") from e
            raise ProviderError(f"Grok couldn't create the image: {details or code.name}") from e


IMAGERS = {i.id: i for i in (GeminiImager(), GrokImager())}


def _slug(text, words=6):
    return "-".join(re.findall(r"[a-z0-9]+", text.lower())[:words]) or "image"


class ImageStudio:
    def __init__(self):
        self.lock = threading.Lock()
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                self.settings = json.load(f)
        except (OSError, ValueError):
            self.settings = {}
        self.settings.setdefault("daily_cap", DEFAULT_DAILY_CAP)
        self.settings.setdefault("main_auto", True)
        self.settings.setdefault("used", {})
        os.makedirs(IMAGES_DIR, exist_ok=True)

    def _save(self):
        tmp = SETTINGS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, indent=2)
        os.replace(tmp, SETTINGS_PATH)

    @property
    def imager(self):
        """The chosen service if its key is set, else whichever image service has a key."""
        chosen = IMAGERS.get(self.settings.get("provider"))
        if chosen and chosen.available():
            return chosen
        return next((i for i in IMAGERS.values() if i.available()), None)

    def used_today(self):
        return self.settings["used"].get(datetime.date.today().isoformat(), 0)

    def can_auto(self):
        return self.imager is not None and self.used_today() < self.settings["daily_cap"]

    def status(self):
        imager = self.imager
        return {
            "provider": imager.id if imager else None,
            "providers": [{"id": i.id, "label": i.label, "available": i.available()} for i in IMAGERS.values()],
            "daily_cap": self.settings["daily_cap"],
            "used_today": self.used_today(),
            "main_auto": self.settings["main_auto"],
        }

    def update(self, provider=None, daily_cap=None, main_auto=None):
        with self.lock:
            if provider is not None:
                if provider not in IMAGERS:
                    raise ValueError(f"Unknown image service '{provider}'.")
                if not IMAGERS[provider].available():
                    raise ValueError(f"No API key for {IMAGERS[provider].label}.")
                self.settings["provider"] = provider
            if daily_cap is not None:
                self.settings["daily_cap"] = max(0, min(int(daily_cap), 500))
            if main_auto is not None:
                self.settings["main_auto"] = bool(main_auto)
            self._save()
        return self.status()

    def create(self, prompt, auto=False):
        """Generates and saves one image. auto=True counts against the daily cap.
        prompt must already be masked. Returns {"file", "url", "provider", "model"}."""
        imager = self.imager
        if imager is None:
            raise ProviderError("Images need a Gemini or Grok key (Claude can't create images).")
        if auto:
            with self.lock:
                if self.used_today() >= self.settings["daily_cap"]:
                    raise ProviderError("The daily limit for images bots create on their own has been reached.")
                today = datetime.date.today().isoformat()
                self.settings["used"] = {today: self.used_today() + 1}  # older days aren't needed
                self._save()
        started = time.time()
        try:
            data, mime, model = imager.create(prompt)
        except Exception:
            if auto:  # a failed attempt shouldn't use up the day's allowance
                with self.lock:
                    today = datetime.date.today().isoformat()
                    self.settings["used"] = {today: max(0, self.used_today() - 1)}
                    self._save()
            raise
        ext = {"image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime, ".png")
        name = f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{_slug(prompt)}{ext}"
        with open(os.path.join(IMAGES_DIR, name), "wb") as f:
            f.write(data)
        print(f"\n[IMAGES]: Created {name} via {imager.label} ({model}) in {time.time() - started:.1f}s")
        return {"file": name, "url": f"/images/{name}", "provider": imager.label, "model": model}


_studio = None
_studio_lock = threading.Lock()


def get_studio():
    global _studio
    with _studio_lock:
        if _studio is None:
            _studio = ImageStudio()
        return _studio
