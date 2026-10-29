import json
import os
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from core.vector_engine import get_vault, SHARED

MEMORY_EXTENSIONS = ('.txt', '.md', '.log', '.py')
MAX_FILE_BYTES = 5 * 1024 * 1024
SKIP_DIRS = {'node_modules', '__pycache__', 'venv', 'site-packages'}

VAULT_DIR = "storage/vault"
FOLDERS_CONFIG = "config/watch_folders.json"
LEDGER_PATH = "storage/ingested.json"  # path -> mtime of the version already learned

_daemon = None


def get_daemon():
    return _daemon


def _is_skipped(path):
    parts = os.path.normpath(path).split(os.sep)
    return any(p.startswith('.') or p in SKIP_DIRS for p in parts[:-1])


class MemoryFileHandler(FileSystemEventHandler):
    def __init__(self, daemon):
        self.daemon = daemon

    def on_created(self, event):
        if not event.is_directory:
            time.sleep(1)  # let the writer finish
            self.daemon.ingest(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.daemon.ingest(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self.daemon.forget(event.src_path)
            self.daemon.ingest(event.dest_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self.daemon.forget(event.src_path)


class MemoryDaemon:
    def __init__(self):
        self.vault = get_vault()
        self.observer = Observer()
        self.handler = MemoryFileHandler(self)
        self.watches = {}
        self.lock = threading.Lock()
        self.ledger = self._load_json(LEDGER_PATH, {})
        self.vault_dir = os.path.abspath(VAULT_DIR)
        # Each extra folder feeds one memory partition: SHARED or a bot's id
        self.extra_folders, self.owners = [], {}
        for entry in self._load_json(FOLDERS_CONFIG, {"folders": []}).get("folders", []):
            path, owner = (entry, SHARED) if isinstance(entry, str) else (entry["path"], entry.get("bot", SHARED))
            self.extra_folders.append(path)
            self.owners[path] = owner

    def _save_config(self):
        self._save_json(FOLDERS_CONFIG, {
            "folders": [{"path": f, "bot": self.owners.get(f, SHARED)} for f in self.extra_folders]
        })

    def owner_of_folder(self, folder):
        return self.owners.get(folder, SHARED)

    def owner_for(self, path):
        """Partition a file belongs to: the owner of the innermost watched folder containing it."""
        path = os.path.normcase(path)
        best, owner = -1, SHARED
        for folder in self.folders():
            prefix = os.path.normcase(folder).rstrip(os.sep) + os.sep
            if path.startswith(prefix) and len(prefix) > best:
                best, owner = len(prefix), self.owner_of_folder(folder)
        return owner

    def folders_of(self, bot_id):
        return [f for f in self.extra_folders if self.owners.get(f) == bot_id]

    @staticmethod
    def _load_json(path, default):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return default

    @staticmethod
    def _save_json(path, data):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def folders(self):
        return [self.vault_dir] + self.extra_folders

    def start(self):
        os.makedirs(self.vault_dir, exist_ok=True)
        self.observer.start()
        for folder in self.folders():
            if os.path.isdir(folder):
                self._watch(folder)
                print(f"[DAEMON]: Active. Monitoring folder: '{folder}'...")
            else:
                print(f"[DAEMON]: Skipping missing folder: '{folder}'")
        # Catch up on anything added while offline, without holding up startup
        threading.Thread(target=self._scan_all, daemon=True).start()

    def stop(self):
        self.observer.stop()

    def join(self):
        self.observer.join()

    def _watch(self, folder):
        self.watches[folder] = self.observer.schedule(self.handler, path=folder, recursive=True)

    def _scan_all(self):
        removed = self.prune()
        if removed:
            print(f"\n[DAEMON]: Cleaned up memories of {removed} file(s) that were deleted or are no longer watched")
        for folder in self.folders():
            self._scan(folder)

    def _is_watched(self, path):
        path = os.path.normcase(path)
        return any(path.startswith(os.path.normcase(f).rstrip(os.sep) + os.sep) for f in self.folders())

    def prune(self):
        """Forgets files that no longer exist or sit outside every watched folder. Returns the count."""
        removed = 0
        for source in self.vault.list_sources():
            if not source.startswith("watcher:"):
                continue
            path = source[len("watcher:"):]
            if not os.path.isfile(path) or not self._is_watched(path):
                self.vault.remove_source(source)
                with self.lock:
                    self.ledger.pop(path, None)
                removed += 1
        if removed:
            with self.lock:
                self._save_json(LEDGER_PATH, self.ledger)
        return removed

    def _scan(self, folder):
        learned = 0
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS]
            for name in files:
                if self.ingest(os.path.join(root, name), quiet=True):
                    learned += 1
        if learned:
            print(f"\n[DAEMON]: Learned {learned} new or changed file(s) from '{folder}'")

    @staticmethod
    def normalize_folder(path):
        folder = os.path.abspath(os.path.expandvars(os.path.expanduser(path.strip().strip('"'))))
        if not os.path.isdir(folder):
            raise ValueError(f"Folder not found: {folder}")
        return folder

    def add_folder(self, path, bot=SHARED):
        """Watches a folder whose files feed the given partition (SHARED or a bot id)."""
        folder = self.normalize_folder(path)
        with self.lock:
            if folder in self.folders():
                raise ValueError("That folder is already being watched.")
            self.extra_folders.append(folder)
            self.owners[folder] = bot
            self._save_config()
            self._watch(folder)
        print(f"\n[DAEMON]: Now monitoring folder: '{folder}' for {'shared memory' if bot == SHARED else 'bot ' + bot}")
        # Files already learned through another watched folder move to their new owner without re-reading
        self._reassign_owners()
        threading.Thread(target=self._scan, args=(folder,), daemon=True).start()
        return folder

    def set_folder_owner(self, folder, bot):
        with self.lock:
            if folder not in self.extra_folders:
                raise ValueError("That folder isn't in the watch list.")
            self.owners[folder] = bot
            self._save_config()
        self._reassign_owners()

    def _reassign_owners(self):
        for source in self.vault.list_sources():
            if source.startswith("watcher:"):
                path = source[len("watcher:"):]
                if self._is_watched(path):
                    self.vault.set_source_owner(source, self.owner_for(path))

    def remove_folder(self, folder, forget=True):
        """Stops watching a folder. forget=False keeps its memories, relabelled so cleanup leaves them alone."""
        with self.lock:
            if folder not in self.extra_folders:
                raise ValueError("That folder isn't in the watch list.")
            self.extra_folders.remove(folder)
            self.owners.pop(folder, None)
            self._save_config()
            watch = self.watches.pop(folder, None)
            if watch:
                self.observer.unschedule(watch)
        # Files still covered by an outer watched folder now belong to that folder's owner
        self._reassign_owners()
        if not forget:
            kept = 0
            for source in self.vault.list_sources():
                path = source[len("watcher:"):] if source.startswith("watcher:") else None
                if path and not self._is_watched(path):
                    self.vault.retag_source(source, f"kept:{path}")
                    with self.lock:
                        self.ledger.pop(path, None)
                    kept += 1
            with self.lock:
                self._save_json(LEDGER_PATH, self.ledger)
            print(f"\n[DAEMON]: Stopped monitoring folder: '{folder}' (kept memories from {kept} file(s))")
            return kept
        removed = self.prune()
        print(f"\n[DAEMON]: Stopped monitoring folder: '{folder}' and forgot {removed} file(s) learned from it")
        return removed

    def remove_bot(self, bot_id):
        """Stops watching a bot's folders and deletes its whole private partition (files are untouched)."""
        with self.lock:
            for folder in self.folders_of(bot_id):
                self.extra_folders.remove(folder)
                self.owners.pop(folder, None)
                watch = self.watches.pop(folder, None)
                if watch:
                    self.observer.unschedule(watch)
                prefix = os.path.normcase(folder).rstrip(os.sep) + os.sep
                for path in [p for p in self.ledger if os.path.normcase(p).startswith(prefix)]:
                    self.ledger.pop(path)
            self._save_config()
            self._save_json(LEDGER_PATH, self.ledger)
        self.vault.remove_bot(bot_id)
        # Anything still covered by another watched folder gets re-learned under that folder's owner
        threading.Thread(target=self._scan_all, daemon=True).start()

    def ingest(self, path, quiet=False):
        """Learns a file if it's new or changed since last time. Returns True if it was learned."""
        path = os.path.abspath(path)
        if not path.lower().endswith(MEMORY_EXTENSIONS) or _is_skipped(path):
            return False
        try:
            stat = os.stat(path)
            if stat.st_size > MAX_FILE_BYTES:
                if not quiet:
                    print(f"\n[DAEMON]: Skipped {os.path.basename(path)} (over 5 MB)")
                return False
            if self.ledger.get(path) == stat.st_mtime:
                return False
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except OSError as e:
            if not quiet:
                print(f"\n[DAEMON]: Could not read {path}: {e}")
            return False

        source = f"watcher:{path}"
        self.vault.remove_source(f"kept:{path}")  # folder re-added: replace kept copy instead of duplicating
        if content.strip():
            self.vault.add_memory(content, metadata={"source": source, "bot_id": self.owner_for(path)})
        else:
            self.vault.remove_source(source)
        with self.lock:
            self.ledger[path] = stat.st_mtime
            self._save_json(LEDGER_PATH, self.ledger)
        if not quiet:
            print(f"\n[DAEMON]: Automatically ingested new neuron node from: {os.path.basename(path)}")
        return True

    def forget(self, path):
        path = os.path.abspath(path)
        if not path.lower().endswith(MEMORY_EXTENSIONS):
            return
        self.vault.remove_source(f"watcher:{path}")
        with self.lock:
            known = self.ledger.pop(path, None) is not None
            if known:
                self._save_json(LEDGER_PATH, self.ledger)
        if known:
            print(f"\n[DAEMON]: Removed memories of deleted file: {os.path.basename(path)}")


def start_memory_daemon():
    global _daemon
    _daemon = MemoryDaemon()
    _daemon.start()
    return _daemon
