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
    "The user is asking about music. First reply in your own words (a sentence or two, in character), then "
    "recommend songs by adding one line per song in exactly this form: "
    "[[SONG: Artist - Song title | one sentence on why]]. Up to 3 songs, only real, released songs, and not "
    "songs you have shared before. Never mention or explain this syntax, and never write links."
)
# The user actually asking for music (this reply may include songs even during the cooldown). Talking ABOUT
# music ("I love music and imagery") is not a request.
MUSIC_REQUEST = re.compile(
    r"\b(recommend|suggest|share|play|send|give|find|pick|show)\w*\b[^.?!]{0,40}?\b(songs?|music|tracks?|playlists?|"
    r"albums?|something to listen)\b"
    r"|\b(any|some|a few|your favou?rite)\s+(songs?|tracks?|music)\b"
    r"|\bwhat (songs?|music|should i (listen|play))\b"
    r"|\b(make|build) (me )?a playlist\b", re.I)


RECOMMEND = re.compile(r"\b(recommend\w*|suggest\w*|ideas for|open (to|for)|more like (that|this|those))\b", re.I)
MUSIC_TOPIC = re.compile(r"\b(songs?|music|tracks?|artists?|albums?|listen\w*|playlists?|genres?|bands?)\b", re.I)
NO_SONG_TOOL_NOTE = ("If you mention a song, just name it in words (\"Title\" by Artist). Never write links, "
                     "\"[Link to ...]\" placeholders or image markup.")
# What small models write when they want to share a song but can't: "![Link to Bon Iver - Holocene]"
_PLACEHOLDER = re.compile(r"!?\[\s*(?:link|listen|play)\w*(?:\s+(?:to|here|on))?\s*([^\]]*)\](?:\s*\((?!https?://)[^)]*\))?", re.I)


def wants_music(message, context=""):
    """The user is asking for music: explicitly, or by asking for recommendations while music is the topic."""
    if MUSIC_REQUEST.search(message):
        return True
    return bool(RECOMMEND.search(message) and MUSIC_TOPIC.search(f"{context[-1500:]} {message}"))


def strip_placeholders(text):
    """'![Link to Bon Iver - Holocene]' becomes 'Bon Iver - Holocene'; a placeholder that names no song
    ('![Link to contemporary track]') disappears. Real web links are left alone."""
    def replace(m):
        inner = m.group(1).strip(" :-")
        return inner if re.search(r"\S\s+(-|–|—|by)\s+\S", inner) else ""
    cleaned = _PLACEHOLDER.sub(replace, text)
    return re.sub(r"\n\s*\n(\s*\n)+", "\n\n", cleaned).strip()


def inline_name(query):
    """'Artist - Title' as it should read inside a sentence: “Title” by Artist."""
    artist, title = _split(query)
    return f"“{title}” by {artist}" if artist else f"“{title}”"


def strip_markers(text):
    """Removes song lines; a song named mid-sentence stays readable as “Title” by Artist."""
    def replace(m):
        start = text.rfind("\n", 0, m.start()) + 1
        end = text.find("\n", m.end())
        line_rest = text[start:m.start()] + text[m.end():end if end >= 0 else len(text)]
        return "" if not line_rest.strip() else inline_name(m.group(1))
    return re.sub(r"\n{3,}", "\n\n", SONG_MARKER.sub(replace, text))
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
    best = min(results, key=lambda x: (bool(_VARIANT.search(x.get("trackName", "") + " " + x.get("collectionName", ""))),
                                       _norm(x.get("artistName", "")) != _norm(artist)))
    return {
        "title": best.get("trackName", title), "artist": best.get("artistName", artist),
        "album": best.get("collectionName", ""),
        "artwork": (best.get("artworkUrl100") or "").replace("100x100bb", "300x300bb") or None,
        "preview": best.get("previewUrl"), "apple": best.get("trackViewUrl"), "found": "iTunes",
    }


_VARIANT = re.compile(r"remix|live|instrumental|karaoke|acoustic|cover|tribute|demo|emulation|edit\b", re.I)


def _deezer(artist, title):
    # The precise artist/track query sometimes returns nothing; a plain search usually finds the original
    queries = [f'artist:"{artist}" track:"{title}"', f"{artist} {title}"] if artist else [title]
    for q in queries:
        data = httpx.get("https://api.deezer.com/search", params={"q": q, "limit": 10}, timeout=TIMEOUT).json().get("data", [])
        found = [x for x in data if _matches(x.get("title", ""), x.get("artist", {}).get("name", ""), title, artist)]
        if found:
            break
    else:
        return None
    t = min(found, key=lambda x: (bool(_VARIANT.search(x.get("title", ""))),
                                  _norm(x.get("artist", {}).get("name", "")) != _norm(artist)))
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
