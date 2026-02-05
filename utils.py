import re

# Utility function to create a safe filename from a blog title
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
