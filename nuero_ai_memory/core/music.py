"""Song lookup for the songs bots share. Uses the free, keyless iTunes Search API (artwork + official
30-second preview), falling back to Deezer's public search. Spotify and YouTube are plain search links
that open in the user's own app, so no Spotify developer access is needed."""
import re
from urllib.parse import quote, quote_plus

import httpx

# Bots add one line per song: [[SONG: Artist - Title | why it fits]]
SONG_MARKER = re.compile(r"\[\[\s*SONG\s*:\s*(.+?)\s*(?:\|\s*(.+?)\s*)?\]\]", re.I | re.S)
MAX_SONGS = 5  # per reply, when the user asked for recommendations
SONG_NOTE = (
    "You may share a song, but only at a genuinely meaningful moment: when something in this conversation truly "
    "moves you or matters deeply to you. That is rare - most conversations should have no song at all, and "
    "ordinary chat never needs one. If, and only if, the moment calls for it, add one line in exactly this form: "
    "[[SONG: Artist - Song title | one sentence on why this song, right now]]. One song at most. Only real, "
    "released songs, never one you have shared before. Never mention or explain this syntax."
)
SONG_REQUEST_NOTE = (
    "The user is asking about music. Recommend songs by adding one line per song in exactly this form: "
    "[[SONG: Artist - Song title | one sentence on why]]. Up to 5 songs, only real, released songs, and not "
    "songs you have shared before. Never mention or explain this syntax."
)
# The user asking for music (this reply may include songs even during the cooldown)
MUSIC_REQUEST = re.compile(r"\b(songs?|music|playlists?|tracks?|album|listen(ing)?|recommend\w* (me )?(a |some )?"
                           r"(song|music|track)|what should i (play|listen))\b", re.I)
TIMEOUT = 6.0


def _links(artist, title):
    # "(Bonus Track)", "[2004 Remaster]" etc. make the streaming services' search worse
    title = re.sub(r"\s*[(\[].*?[)\]]", "", title).split(" - ")[0]
    text = f"{artist} {title}".strip()
    return {
        "spotify": f"https://open.spotify.com/search/{quote(text, safe='')}",
        "youtube": f"https://www.youtube.com/results?search_query={quote_plus(text)}",
    }


def _split(query):
    """'Artist - Title' (or 'Title by Artist') -> (artist, title)."""
    query = query.strip().strip('"“”')
    for sep in (" - ", " – ", " — "):
        if sep in query:
            artist, title = query.split(sep, 1)
            return artist.strip(), title.strip().strip('"“”')
    by = re.match(r"(.+?)\s+by\s+(.+)", query, re.I)
    if by:
        return by.group(2).strip(), by.group(1).strip().strip('"“”')
    return "", query


def _norm(text):
    """'Dreams (2004 Remaster)' -> 'dreams', for comparing titles and artists."""
    text = re.sub(r"\s*[(\[].*?[)\]]", "", text.lower()).split(" - ")[0]
    return re.sub(r"[^a-z0-9]+", " ", text.replace("&", "and")).strip()


def same_song(a, b):
    """True when two song cards are the same song (ignoring remaster tags, punctuation and case)."""
    return _norm(a.get("title", "")) == _norm(b.get("title", "")) and \
        _norm(a.get("artist", "")).split(" and ")[0] == _norm(b.get("artist", "")).split(" and ")[0]


def _matches(found_title, found_artist, title, artist):
    """Only accept the song that was asked for: a wrong song is worse than no artwork."""
    t, ft = _norm(title), _norm(found_title)
    a, fa = _norm(artist), _norm(found_artist)
    title_ok = bool(t) and (ft == t or ft.startswith(t + " ") or t.startswith(ft + " "))
    artist_ok = not a or a in fa or fa in a
    return title_ok and artist_ok


def _itunes(artist, title):
    r = httpx.get("https://itunes.apple.com/search", timeout=TIMEOUT,
                  params={"term": f"{artist} {title}".strip(), "entity": "song", "limit": 10})
    results = [x for x in r.json().get("results", [])
               if _matches(x.get("trackName", ""), x.get("artistName", ""), title, artist)]
    if not results:
        return None
    # Originals beat remixes/live/instrumental versions; then the named artist alone beats duets and features
    variant = re.compile(r"remix|live|instrumental|karaoke|acoustic|cover|tribute|demo|edit\b", re.I)
    best = min(results, key=lambda x: (bool(variant.search(x.get("trackName", "") + " " + x.get("collectionName", ""))),
                                       _norm(x.get("artistName", "")) != _norm(artist)))
    return {
        "title": best.get("trackName", title), "artist": best.get("artistName", artist),
        "album": best.get("collectionName", ""),
        "artwork": (best.get("artworkUrl100") or "").replace("100x100bb", "300x300bb") or None,
        "preview": best.get("previewUrl"), "apple": best.get("trackViewUrl"), "found": "iTunes",
    }


def _deezer(artist, title):
    q = f'artist:"{artist}" track:"{title}"' if artist else title
    data = httpx.get("https://api.deezer.com/search", params={"q": q, "limit": 5}, timeout=TIMEOUT).json().get("data", [])
    t = next((x for x in data if _matches(x.get("title", ""), x.get("artist", {}).get("name", ""), title, artist)), None)
    if not t:
        return None
    return {
        "title": t.get("title", title), "artist": t.get("artist", {}).get("name", artist),
        "album": t.get("album", {}).get("title", ""), "artwork": t.get("album", {}).get("cover_medium"),
        "preview": t.get("preview"), "deezer": t.get("link"), "found": "Deezer",
    }


def lookup(query):
    """A song card for 'Artist - Title'. Works offline too: without a match it still has search links."""
    artist, title = _split(query)
    song = None
    for source in (_itunes, _deezer):
        try:
            song = source(artist, title)
        except (httpx.HTTPError, ValueError):
            song = None
        if song:
            break
    song = song or {"title": title, "artist": artist, "album": "", "artwork": None, "preview": None, "found": None}
    song["links"] = _links(song["artist"], song["title"])
    if song.get("apple"):
        song["links"]["apple"] = song.pop("apple")
    if song.get("deezer"):
        song["links"]["deezer"] = song.pop("deezer")
    return song
