"""Adapters for each AI service. Every provider takes (model, prompt, system, history) and returns reply text,
raising ProviderError with a message that's safe to show in the chat.
history is a list of {"role": "user" | "assistant", "text": ...}, oldest first, not including prompt."""
import os
import re
import time

import httpx

# Dropped connections and timeouts, from whichever HTTP library an SDK picked
NETWORK_ERRORS = (httpx.TransportError,)
TIMEOUT_ERRORS = (httpx.TimeoutException,)
try:
    import httpx2
    NETWORK_ERRORS += (httpx2.TransportError,)
    TIMEOUT_ERRORS += (httpx2.TimeoutException,)
except ImportError:
    pass


class ProviderError(Exception):
    """A failure with a plain-language message for the chat window."""


class ProviderBusyError(ProviderError):
    """The service kept failing with temporary errors (overload, rate limit, timeout)."""


class Provider:
    id = ""
    label = ""
    key_env = ""
    default_model = ""

    def __init__(self):
        self._client = None
        self._models_cache = None

    def has_key(self) -> bool:
        return bool(os.environ.get(self.key_env, "").strip())

    @property
    def missing_label(self):
        """Shown next to the service name when it can't be used."""
        return "no key"

    @property
    def missing_hint(self):
        return f"Set {self.key_env} and restart Nuero AI to use {self.label}."

    def _key(self) -> str:
        if not self.has_key():
            raise ProviderError(f"No API key for {self.label}. Set {self.key_env} and restart Nuero AI.")
        return os.environ[self.key_env].strip()

    def models(self) -> list:
        """[{"id": ..., "label": ...}] for the model picker. Listing is cached after the first success."""
        if self._models_cache is None:
            try:
                listed = self._list_models()
            except Exception:
                listed = []  # offline or bad key: offer the default only, and try again next time
            if listed:
                self._models_cache = listed
            else:
                return [{"id": self.default_model, "label": self.default_model}]
        return self._models_cache

    def _list_models(self) -> list:
        return []

    def generate(self, model: str, prompt: str, system: str, history=None) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------- Gemini

GEMINI_RETRY_PAUSE = 3  # seconds to wait after every model has failed once, before going round again
GEMINI_MAX_ROUNDS = 2  # passes through the model list per reply, so failures don't burn the request quota
GEMINI_STICKY_SECONDS = 10 * 60  # how long to keep using a backup that worked before trying the chosen model again
GEMINI_REQUEST_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "18"))  # seconds one request may take
GEMINI_TOTAL_BUDGET = 60  # seconds for one reply across all retries and backups
# How hard the model "thinks" before answering: minimal, low, medium, high, or off (model default)
GEMINI_THINKING = os.environ.get("GEMINI_THINKING", "low").strip().upper()

# Model names that aren't general-purpose chat models
_GEMINI_NOT_CHAT = ("preview", "exp", "image", "tts", "audio", "live", "embedding", "robotics", "computer-use")


def _gemini_version_key(name: str):
    """'gemini-3.8-flash' -> (3, 8) so newer models sort first."""
    digits = name.split("gemini-")[-1].split("-")[0]
    try:
        return tuple(int(p) for p in digits.split("."))
    except ValueError:
        return (0,)


class GeminiProvider(Provider):
    id = "gemini"
    label = "Gemini"
    key_env = "GEMINI_API_KEY"
    default_model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

    def __init__(self):
        super().__init__()
        # Comma-separated list; None = auto-pick on first need
        env_fallback = os.environ.get("GEMINI_FALLBACK_MODEL")
        self._fallbacks = [m.strip() for m in env_fallback.split(",") if m.strip()] if env_fallback else None
        # After a backup works, keep using it for a while instead of re-waiting on the busy chosen model
        self._sticky = {}  # chosen model -> (backup model, until)
        self._no_thinking = set()  # models that rejected the thinking setting

    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._key())
        return self._client

    def _chat_model_names(self):
        return [
            m.name.removeprefix("models/") for m in self.client().models.list()
            if "generateContent" in (m.supported_actions or [])
            and m.name.removeprefix("models/").startswith("gemini")
            and not any(t in m.name for t in _GEMINI_NOT_CHAT)
        ]

    def _list_models(self):
        names = sorted(self._chat_model_names(), key=lambda n: (tuple(-p for p in _gemini_version_key(n)), n))
        return [{"id": n, "label": n} for n in names]

    def _fallback_models(self, chosen):
        """Up to two backups this key can use, picked from Google's own model list:
        the newest other full Flash, then the newest Flash-Lite (lighter, often less overloaded)."""
        if self._fallbacks is None:
            try:
                names = self._chat_model_names()
            except Exception:
                names = []
            candidates = [n for n in names if "flash" in n]
            newest_first = lambda n: tuple(-p for p in _gemini_version_key(n))
            # Two of each kind, so one remains when the chosen model is itself one of them
            full = sorted((n for n in candidates if "lite" not in n), key=newest_first)[:2]
            lite = sorted((n for n in candidates if "lite" in n), key=newest_first)[:2]
            self._fallbacks = full + lite
            if self._fallbacks:
                print(f"\n[GEMINI]: Backup models available: {', '.join(self._fallbacks)}")
        others = [m for m in self._fallbacks if m != chosen]
        full = [m for m in others if "lite" not in m]
        lite = [m for m in others if "lite" in m]
        return (full[:1] + lite[:1]) or others[:2]

    def _config(self, model, system, timeout):
        config = {
            "system_instruction": system,
            "http_options": {"timeout": int(timeout * 1000)},
        }
        if GEMINI_THINKING not in ("", "OFF") and model not in self._no_thinking:
            config["thinking_config"] = {"thinking_level": GEMINI_THINKING}
        return config

    @staticmethod
    def _is_temporary(e):
        from google.genai import errors
        if isinstance(e, NETWORK_ERRORS):
            return True
        return isinstance(e, errors.APIError) and (e.code == 429 or (e.code or 0) >= 500)

    def generate(self, model, prompt, system, history=None):
        """Calls Gemini, retrying overloads (503/429) and dropped connections,
        then switching to a backup model if the chosen one stays unavailable."""
        from google.genai import errors
        client = self.client()
        contents = [
            {"role": "model" if turn["role"] == "assistant" else "user", "parts": [{"text": turn["text"]}]}
            for turn in history or []
        ] + [{"role": "user", "parts": [{"text": prompt}]}]
        started = time.time()
        deadline = started + GEMINI_TOTAL_BUDGET
        backup, until = self._sticky.get(model, (None, 0.0))
        sticky = backup if time.time() < until else None
        # One attempt per model in turn, so every backup gets a chance before time runs out;
        # if all fail, pause briefly and go round again while budget remains
        models = [sticky or model]
        tried = []
        i = 0
        rounds = 1
        while time.time() < deadline - 3:
            if i == len(models):
                extra = [m for m in [model] + self._fallback_models(model) if m not in models]
                if extra:
                    models += extra
                elif rounds < GEMINI_MAX_ROUNDS:
                    rounds += 1
                    time.sleep(GEMINI_RETRY_PAUSE)
                    i = 0
                    continue
                else:
                    break
            attempt = models[i]
            i += 1
            if attempt not in tried:
                tried.append(attempt)
            timeout = min(GEMINI_REQUEST_TIMEOUT, deadline - time.time())
            if timeout < 3:
                break
            try:
                response = client.models.generate_content(
                    model=attempt,
                    contents=contents,
                    config=self._config(attempt, system, timeout)
                )
                if attempt != model and attempt != sticky:
                    # Don't extend on later successes, so the chosen model gets retried after the window
                    self._sticky[model] = (attempt, time.time() + GEMINI_STICKY_SECONDS)
                elif sticky and attempt == model:
                    self._sticky.pop(model, None)
                print(f"\n[GEMINI]: Replied via {attempt} in {time.time() - started:.1f}s")
                return response.text or ""
            except Exception as e:
                if (isinstance(e, errors.ClientError) and e.code == 400
                        and "thinking" in str(e.message).lower() and attempt not in self._no_thinking):
                    self._no_thinking.add(attempt)  # model doesn't support it; retry right away without
                    i -= 1
                    continue
                if isinstance(e, errors.ClientError) and e.code in (400, 401, 403) and "key" in str(e.message).lower():
                    raise ProviderError("Gemini rejected the API key. Check GEMINI_API_KEY.") from e
                if not self._is_temporary(e):
                    raise
                if isinstance(e, TIMEOUT_ERRORS):
                    detail = f"no answer within {timeout:.0f}s"
                elif isinstance(e, errors.APIError):
                    detail = f"{e.code} {e.status}: {e.message}"
                else:
                    detail = str(e)
                print(f"\n[GEMINI]: {attempt} failed ({detail[:120]}), trying next model...")
        raise ProviderBusyError(
            f"Gemini is busy or unreachable right now (tried {', '.join(tried)} "
            f"over {time.time() - started:.0f}s). Wait a minute, or switch AI in the chat panel."
        )


# ---------------------------------------------------------------- Claude

# (model id, picker label, effort): effort scales with the model so heavier picks also think harder
CLAUDE_MODELS = [
    ("claude-haiku-5-5", "Haiku 5.5 — fast, everyday chat", "low"),
    ("claude-sonnet-5-5", "Sonnet 5.5 — stronger reasoning", "medium"),
    ("claude-opus-5-5", "Opus 5.5 — heavy lifting", "high"),
]
# Server-side refusal fallback; the "default" form is available for these models on the Claude API
CLAUDE_FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5"}


class ClaudeProvider(Provider):
    id = "claude"
    label = "Claude"
    key_env = "ANTHROPIC_API_KEY"
    default_model = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")

    def client(self):
        if self._client is None:
            import anthropic
            # The SDK retries connection errors, 429 and 5xx with backoff on its own
            self._client = anthropic.Anthropic(api_key=self._key(), timeout=180.0, max_retries=2)
        return self._client

    def models(self):
        return [{"id": m, "label": label} for m, label, _ in CLAUDE_MODELS]

    def generate(self, model, prompt, system, history=None):
        import anthropic
        client = self.client()
        effort = next((e for m, _, e in CLAUDE_MODELS if m == model), "low")
        messages = [{"role": turn["role"], "content": turn["text"]} for turn in history or []]
        while messages and messages[0]["role"] != "user":
            messages.pop(0)  # the conversation must open with a user turn
        request = dict(
            model=model,
            max_tokens=16000,
            system=system,
            messages=messages + [{"role": "user", "content": prompt}],
            output_config={"effort": effort},
        )
        started = time.time()
        try:
            if model in CLAUDE_FALLBACK_MODELS:
                response = client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
                )
            else:
                response = client.messages.create(**request)
        except anthropic.AuthenticationError as e:
            raise ProviderError("Claude rejected the API key. Check ANTHROPIC_API_KEY.") from e
        except anthropic.PermissionDeniedError as e:
            raise ProviderError(f"Claude says this key can't use {model}: {e.message}") from e
        except anthropic.NotFoundError as e:
            raise ProviderError(f"Claude doesn't recognise the model {model}.") from e
        except anthropic.BadRequestError as e:
            if "credit" in str(e.message).lower():
                raise ProviderError("Your Claude credit balance is too low. Add credits in the Claude Console.") from e
            raise ProviderError(f"Claude couldn't handle that request: {e.message}") from e
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            # APIConnectionError covers timeouts; the SDK has already retried by this point
            raise ProviderBusyError("Claude is busy or unreachable right now. Wait a minute, or switch AI.") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise ProviderBusyError("Claude is busy right now. Wait a minute, or switch AI.") from e
            raise ProviderError(f"Claude error {e.status_code}: {e.message}") from e

        if response.stop_reason == "refusal":
            raise ProviderError("Claude declined to answer that one. Try rephrasing it.")
        text = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason == "max_tokens":
            text += "\n\n[Reply cut off at the length limit.]"
        print(f"\n[CLAUDE]: Replied via {response.model} in {time.time() - started:.1f}s")
        return text


# ---------------------------------------------------------------- Grok

GROK_TIMEOUT = 120.0
# How hard Grok reasons before answering: none, low, medium, high, or off (model default)
GROK_REASONING = os.environ.get("GROK_REASONING", "low").strip().lower()
# Listed models that aren't for plain chat
_GROK_NOT_CHAT = ("imagine", "image", "video", "code", "build", "multi-agent", "embed")


class GrokProvider(Provider):
    id = "grok"
    label = "Grok"
    key_env = "XAI_API_KEY"
    default_model = os.environ.get("GROK_MODEL", "grok-4.7")

    def __init__(self):
        super().__init__()
        self._no_reasoning = set()  # models that rejected reasoning_effort

    def client(self):
        if self._client is None:
            from xai_sdk import Client
            self._client = Client(api_key=self._key(), timeout=GROK_TIMEOUT)
        return self._client

    def _list_models(self):
        names = [m.name for m in self.client().models.list_language_models()
                 if not any(t in m.name for t in _GROK_NOT_CHAT)]
        if self.default_model not in names:
            names.append(self.default_model)
        return [{"id": n, "label": n} for n in sorted(names, reverse=True)]

    def generate(self, model, prompt, system, history=None):
        import grpc
        from xai_sdk.chat import system as system_message, user as user_message, assistant as assistant_message
        client = self.client()
        started = time.time()
        while True:
            options = {}
            if GROK_REASONING not in ("", "off") and model not in self._no_reasoning:
                options["reasoning_effort"] = GROK_REASONING
            try:
                chat = client.chat.create(model=model, messages=[system_message(system)], **options)
                for turn in history or []:
                    chat.append((assistant_message if turn["role"] == "assistant" else user_message)(turn["text"]))
                chat.append(user_message(prompt))
                response = chat.sample()
                print(f"\n[GROK]: Replied via {model} in {time.time() - started:.1f}s")
                return response.content or ""
            except grpc.RpcError as e:
                code, details = e.code(), str(e.details() or "")
                if (code == grpc.StatusCode.INVALID_ARGUMENT and "reason" in details.lower()
                        and model not in self._no_reasoning):
                    self._no_reasoning.add(model)  # model doesn't support it; retry right away without
                    continue
                if code in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED) or "api key" in details.lower():
                    raise ProviderError("Grok rejected the API key. Check XAI_API_KEY.") from e
                if code == grpc.StatusCode.NOT_FOUND:
                    raise ProviderError(f"Grok doesn't recognise the model {model}.") from e
                if code in (grpc.StatusCode.RESOURCE_EXHAUSTED, grpc.StatusCode.UNAVAILABLE,
                            grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.INTERNAL):
                    raise ProviderBusyError("Grok is busy or unreachable right now. Wait a minute, or switch AI.") from e
                raise ProviderError(f"Grok error: {details or code.name}") from e


# ---------------------------------------------------------------- Ollama (local)

def _ollama_host():
    """OLLAMA_HOST as Ollama itself reads it ("0.0.0.0:11434", "localhost", "http://…"), as a URL to call."""
    host = os.environ.get("OLLAMA_HOST", "").strip() or "127.0.0.1:11434"
    if "://" not in host:
        host = "http://" + host
    scheme, rest = host.split("://", 1)
    rest = rest.replace("0.0.0.0", "127.0.0.1")
    if ":" not in rest.split("/")[0]:
        rest = rest.split("/")[0] + ":11434"
    return f"{scheme}://{rest}".rstrip("/")


class OllamaProvider(Provider):
    """Models running on this computer through Ollama: nothing leaves the machine, works offline."""
    id = "ollama"
    label = "Ollama (local)"
    key_env = "OLLAMA_HOST"
    default_model = os.environ.get("OLLAMA_MODEL", "")
    TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))  # local models on modest hardware can be slow

    def __init__(self):
        super().__init__()
        self._up, self._checked = False, 0.0

    def has_key(self):
        """'Available' means the Ollama server answers; checked at most every 10 seconds."""
        if time.time() - self._checked > 10:
            try:
                self._up = httpx.get(f"{_ollama_host()}/api/version", timeout=1.5).status_code == 200
            except httpx.HTTPError:
                self._up = False
            self._checked = time.time()
        return self._up

    @property
    def missing_label(self):
        return "not running"

    @property
    def missing_hint(self):
        return "Start Ollama on this computer, then pick it again."

    def models(self):
        # Always re-list, so models pulled while Nuero AI runs show up
        try:
            names = [m["name"] for m in httpx.get(f"{_ollama_host()}/api/tags", timeout=3).json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            names = []
        if names and not self.default_model:
            self.default_model = names[0]
        return [{"id": n, "label": n} for n in names] or [{"id": self.default_model or "llama3.2", "label": self.default_model or "llama3.2"}]

    MAX_CONTEXT = int(os.environ.get("OLLAMA_MAX_CTX", "32768"))

    def _context_size(self, messages):
        """Ollama's default window is small and it silently drops the start of longer prompts - which is
        the personality, memories and attached books. Size it to fit, in a few fixed steps, since changing
        the size makes Ollama reload the model."""
        needed = sum(len(m["content"]) for m in messages) // 3 + 1500  # ~3 characters per token, plus reply room
        size = 8192
        while size < needed and size < self.MAX_CONTEXT:
            size *= 2
        return min(size, self.MAX_CONTEXT)

    def generate(self, model, prompt, system, history=None):
        model = model or self.default_model or self.models()[0]["id"]
        messages = [{"role": "system", "content": system}]
        messages += [{"role": turn["role"], "content": turn["text"]} for turn in history or []]
        messages.append({"role": "user", "content": prompt})
        started = time.time()
        try:
            response = httpx.post(f"{_ollama_host()}/api/chat", timeout=self.TIMEOUT,
                                  json={"model": model, "messages": messages, "stream": False,
                                        "options": {"num_ctx": self._context_size(messages)}})
        except httpx.ConnectError as e:
            raise ProviderError("Ollama isn't running. Start it on this computer, or switch AI.") from e
        except httpx.TimeoutException as e:
            raise ProviderBusyError(f"{model} took longer than {self.TIMEOUT:.0f}s to answer. "
                                    "Try a smaller model, or set OLLAMA_TIMEOUT higher.") from e
        if response.status_code == 404:
            raise ProviderError(f"Ollama doesn't have the model {model}. Install it with: ollama pull {model}")
        if response.status_code >= 400:
            try:
                detail = response.json().get("error", response.text)
            except ValueError:
                detail = response.text
            raise ProviderError(f"Ollama error: {detail}")
        text = response.json().get("message", {}).get("content", "")
        # Reasoning models may include their thinking inline; only the answer belongs in the chat
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        print(f"\n[OLLAMA]: Replied via {model} in {time.time() - started:.1f}s")
        return text


PROVIDERS = {p.id: p for p in (GeminiProvider(), ClaudeProvider(), GrokProvider(), OllamaProvider())}
