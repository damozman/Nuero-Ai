import os
import threading

# Every module uses project-relative paths, so run from the project root regardless of launch dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from daemons.watcher import start_memory_daemon
from core.dashboard import run_dashboard
from core.pipeline import get_pipeline
from core.providers import PROVIDERS
from core import profile


def show_ai_options(pipeline):
    status = pipeline.ai_status()
    print(f"\n  Using: {pipeline.describe()}")
    for p in status["providers"]:
        if p["available"]:
            print(f"  {p['id']:7} models: {', '.join(m['id'] for m in p['models'])}")
        else:
            print(f"  {p['id']:7} ({p['missing_label']} - {p['missing_hint']})")
    print("  Switch with e.g.  /ai claude sonnet   /ai grok   /ai ollama llama   /ai gemini flash-lite\n")


def show_bots(pipeline, current):
    bots = pipeline.bots.bots
    helper = f"{profile.system_name()} (system helper)"
    print(f"\n  Talking to: {pipeline.bots.get(current)['name'] if current else helper}")
    print("  Family Nodes: " + (", ".join(b["name"] for b in bots) if bots
                                else "none yet - add them in the dashboard's Family Nodes panel"))
    print("  Switch with  /node <name>   back to the system helper with  /node none"
          "   ask every Family Node with  /council <message>\n")


def run_cli_chat():
    pipeline = get_pipeline()
    current_bot = None  # None = the system helper
    histories = {}  # bot id (or None) -> this session's turns, so each conversation has context
    council_log = []  # the council's shared conversation this session: [{"speaker", "text"}]
    print("\n" + "="*50)
    print(f"  {profile.system_name().upper()} IS RUNNING")
    print(f"  Interactive Chat Mode Connected to {pipeline.describe()}")
    print("  Type /ai to switch AI services and models, /node to talk to a Family Node, /council to ask them all,")
    print("  /image <description> to create a picture")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ")
            command = user_input.strip().lower()
            if command in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
            if command.startswith("/ai"):
                parts = user_input.split()[1:]
                if parts:
                    print(f"\n  Now using {pipeline.set_ai(parts[0], ' '.join(parts[1:]) or None)}\n")
                else:
                    show_ai_options(pipeline)
                continue
            if command.startswith(("/node", "/bot")):  # /bot still works as the old name
                hint = user_input.strip().split(maxsplit=1)[1].strip() if " " in user_input.strip() else ""
                if not hint:
                    show_bots(pipeline, current_bot)
                elif hint.lower() in ("none", "main", "system", profile.system_name().lower()):
                    current_bot = None
                    print(f"\n  Now talking to {profile.system_name()} (system helper)\n")
                else:
                    bot = pipeline.bots.find(hint)
                    if bot:
                        current_bot = bot["id"]
                        print(f"\n  Now talking to {bot['name']}\n")
                    else:
                        print(f"\n  No Family Node matching '{hint}'.\n")
                        show_bots(pipeline, current_bot)
                continue
            if command.startswith("/image"):
                prompt = user_input.strip()[6:].strip()
                if not prompt:
                    print("\n  Usage: /image <description of the picture>\n")
                    continue
                image = pipeline.create_image(prompt, current_bot)
                print(f"\n  Image saved: {os.path.abspath(os.path.join('storage', 'images', image['file']))}")
                print(f"  (also visible in the dashboard at http://127.0.0.1:8050{image['url']})\n")
                continue
            if command.startswith("/council"):
                question = user_input.strip()[8:].strip()
                if not question:
                    print("\n  Usage: /council <message>\n")
                    continue
                replies = []
                for result in pipeline.council(question, transcript=council_log):
                    if "error" in result:
                        print(f"\n[{result['bot_name']}] (failed) {result['error']}")
                    else:
                        print(f"\n[{result['bot_name']}]: {result['reply']}")
                        replies.append({"speaker": result["bot_name"], "text": result["reply"]})
                council_log += [{"speaker": "user", "text": question}] + replies
                print("-"*50 + "\n")
                continue

            history = histories.setdefault(current_bot, [])
            result = pipeline.respond(user_input, current_bot, history=history)
            response = result["reply"]
            history += [{"role": "user", "text": user_input}, {"role": "assistant", "text": response}]
            name = pipeline.bots.get(current_bot)["name"] if current_bot else "AI"
            print(f"\n{name}: {response}")
            for image in result["images"]:
                print(f"\n  [{name} created an image: {image['prompt']}]")
                print(f"  Saved: {os.path.abspath(os.path.join('storage', 'images', image['file']))}")
            if result["image_error"]:
                print(f"\n  [{result['image_error']}]")
            for item in result["archived"]:
                print(f"\n  [{name} archived \"{item['title']}\" in {item['book_title']}]")
            print("-"*50 + "\n")
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"\n[ERROR]: {e}\n")


if __name__ == "__main__":
    if not any(p.has_key() for p in PROVIDERS.values()):
        keys = ", ".join(p.key_env for p in PROVIDERS.values() if p.id != "ollama")
        print(f"[WARN]: No AI service is available. Set a key ({keys}) or start Ollama; chat fails until then.")

    # 1. Establish structural directories before booting daemons
    os.makedirs("storage/vault", exist_ok=True)
    os.makedirs("config", exist_ok=True)

    # Pre-seed fallback baseline identity if missing
    if not os.path.exists("config/identity_frequency.txt"):
        with open("config/identity_frequency.txt", "w", encoding="utf-8") as f:
            f.write("You are {system_name}, the system helper for this household's AI app.")

    # 2. Spin up File-System Watcher as an isolated background task
    try:
        daemon = start_memory_daemon()
    except RuntimeError as e:
        if "already accessed" not in str(e):
            raise
        print(f"[ERROR]: {profile.system_name()} is already running in another terminal (the memory vault is locked).")
        print("         Type 'exit' in that window, or close it, then start again.")
        raise SystemExit(1)

    # 3. Spin up WebGL Visualization Server in a concurrent execution thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    print("[DASHBOARD]: Interactive WebGL interface hosted online at http://127.0.0.1:8050")

    # 4. Trigger Interactive Control Panel Terminal Loop directly
    run_cli_chat()

    # Clean termination sequence
    daemon.stop()
    daemon.join()
