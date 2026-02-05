import random
import sys
import threading
import time
import re

LOADING_MESSAGES = [
    "🧠 Planning blog structure...",
    "✍️ Writing sections in parallel...",
    "📚 Adding real-world insights...",
    "🧩 Stitching everything together...",
    "✨ Polishing the final blog...",
    "🚀 Almost there..."
]


def loading_animation(stop_event: threading.Event):
    spinner = ["|", "/", "-", "\\"]
    idx = 0

    while not stop_event.is_set():
        message = random.choice(LOADING_MESSAGES)
        sys.stdout.write(f"\r{spinner[idx % len(spinner)]} {message}")
        sys.stdout.flush()

        idx += 1
        time.sleep(0.6)

    # Clear line after stop
    sys.stdout.write("\r✅ Blog generation complete!            \n")
    sys.stdout.flush()


def safe_filename(title: str) -> str:
    # lower case
    name = title.lower()

    # replace spaces with underscore
    name = name.replace(" ", "_")

    # remove anything that's NOT a-z, 0-9, _ or -
    name = re.sub(r"[^a-z0-9_-]", "", name)

    # avoid empty names
    if not name:
        name = "blog"

    return f"{name}.md"
