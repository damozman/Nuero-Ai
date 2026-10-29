"""The built-in 'Nuero AI Guide' book: how the app works, so any AI it's attached to can answer questions
about Nuero AI itself. Contains no personal data. Bump GUIDE_VERSION when the text changes."""

GUIDE_VERSION = 2
GUIDE_TITLE = "Nuero AI Guide"

GUIDE_PAGES = [
    ("What Nuero AI is",
     "Nuero AI is a personal AI app running on the user's own computer, opened in the browser at "
     "http://127.0.0.1:8050. The AI personalities in it are called Family Nodes: each has its own personality "
     "and private memory. There is also a Council mode, a Library, images, music and voice. Memories are stored "
     "on this computer. The app is started with start_nuero.bat (or '.\\start_nuero local' to use the local AI)."),
    ("Who can answer",
     "The left sidebar lists the Family Nodes first, then Council (several Family Nodes answer the same message "
     "at once, each from its own memory; Members chooses who is in). At the bottom, under System, is Nuero AI: "
     "the system helper, for questions about how the app works. Each conversation is kept separately, and the AI "
     "sees the last 10 messages of that conversation. Clear starts a conversation fresh."),
    ("Memory",
     "Before answering, the AI looks up the 3 memories most related to the message. Memories are either shared "
     "(every Family Node can use them) or private to one Family Node. Family Nodes never see each other's private "
     "memories, not even in Council. Ways to add memories: type text and click 'Save as memory' (saves to the "
     "Family Node you are talking to, or shared); drop .txt, .md, .log or .py files into the quick drop folder "
     "(storage/vault, shared); or add watched folders in the Memory panel, choosing Shared or a Family Node. "
     "Edited files are re-learned, deleted files forgotten. Under each reply, 'memories used' shows what was "
     "drawn on, with a Forget button."),
    ("Family Nodes",
     "Family Nodes are created in the Family Nodes panel (button at the bottom of the sidebar, or '+ New'): paste "
     "the path of the Family Node's folder and click 'Create Family Node'. Everything in that folder becomes its "
     "private memory, and a file named persona/personality/identity/instructions sets its starting personality. "
     "Each Family Node card has: name, picture (upload or 'Create' to have the image AI paint one), personality, "
     "switches for creating images and sharing songs on its own, a Voice section (voice, speed, pitch, preview) "
     "and Skills."),
    ("Skills",
     "Skills are named sets of instructions added to a Family Node in its card (written there or imported from a "
     ".md or .txt file), e.g. 'Copywriter' or 'Tarot reader'. In a conversation with that Family Node, the 🧩 "
     "menu in the header switches the active skill, or none."),
    ("Library",
     "The Library (top of the sidebar) is a bookshelf. Shelves: 'Your books' (made by the user, plus this "
     "guide), 'Playlists' (songs each Family Node has shared) and 'Family Node archives' (things each Family "
     "Node decided were important to keep). Pages can be added with '📚 Save to library' under replies, "
     "memories, images and songs, or inside a book with 'Write a page' or 'Add from memories'. A book can be "
     "attached to any conversation ('Use in a conversation', or the 📚 button by the message box); while "
     "attached, the AI reads the whole book with every message."),
    ("Choosing the AI",
     "The AI menu at the bottom-left of the sidebar picks the service and model for every Family Node: Gemini, "
     "Claude, Grok (cloud services, each needs its API key: GEMINI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY) or "
     "Ollama (local). Claude models go Haiku (fast) to Sonnet to Opus (heaviest). Names and addresses are "
     "masked before anything is sent to a cloud service."),
    ("Local AI (Ollama) and privacy",
     "Ollama runs AI models on this computer, so conversations never leave it and it works offline. Start "
     "Ollama, then pick 'Ollama (local)' and an installed model (install more with 'ollama pull <model>'). "
     "While it is selected, Family Nodes do not create images on their own. For full privacy, also avoid the "
     "🎤 microphone (the browser transcribes speech online) and voices marked ⭐ (they stream from Microsoft). "
     "Song cards still look up artwork and previews online when a Family Node shares a song."),
    ("Images",
     "Type a description and click 🎨 (or type /image description) to create a picture. Family Nodes may also "
     "create images on their own when it fits, up to a daily limit set in the Family Nodes panel, never in "
     "Council. Images need a Gemini or Grok key (Claude cannot create images) and are saved in storage/images."),
    ("Music",
     "Family Nodes can share songs they are feeling or recommend music. Each song appears as a card with "
     "artwork, the Family Node's reason, a 30-second preview and buttons to open it in Spotify, YouTube or Apple "
     "Music. Every shared song is collected in that Family Node's playlist on the Playlists shelf. Song details "
     "come from Apple's and Deezer's free public search; the AI cannot listen to audio itself."),
    ("Voice",
     "Click 🎤 to speak a message; it is sent 3 seconds after you stop talking (click 🎤 again to send at once). "
     "'Voice on' reads replies aloud, each Family Node in its own voice. Microsoft Edge has the most natural "
     "free voices (marked ⭐)."),
    ("Terminal commands",
     "In the terminal window: /ai (show or switch AI, e.g. /ai claude sonnet or /ai ollama), /node (list Family "
     "Nodes), /node Name (talk to a Family Node), /node none (back to the system helper), /council question "
     "(ask every Family Node), /image description, and exit to stop the app."),
]
