import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"

# Regex pattern matching any variant of the hardcoded PRAVASH path
PRAVASH_PATTERN = re.compile(
    r'["\'](?:C:\\+Users\\+PRAVASH\\+Desktop\\+NLP\\+PBL\\+AI-Research-Paper-Assistant|C:/Users/PRAVASH/Desktop/NLP/PBL/AI-Research-Paper-Assistant)([^"\']*)["\']',
    re.IGNORECASE
)

def fix_file(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "PRAVASH" not in content:
        return False

    # Compute relative depth from this file to PROJECT_ROOT
    # e.g., backend/src/intelligence/x.py is 4 levels deep
    rel_levels = len(file_path.relative_to(PROJECT_ROOT).parts) - 1
    parents_chain = ".parent" * rel_levels

    # Replacement function to resolve path dynamically from PROJECT_ROOT
    def replacer(match):
        subpath = match.group(1).replace("\\", "/").lstrip("/")
        return f'str(pathlib.Path(__file__).resolve(){parents_chain} / "{subpath}")'

    new_content = PRAVASH_PATTERN.sub(replacer, content)

    # Ensure pathlib is imported
    if "import pathlib" not in new_content and "from pathlib import" not in new_content:
        new_content = "import pathlib\n" + new_content

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"[Fixed] {file_path.relative_to(PROJECT_ROOT)}")
    return True

if __name__ == "__main__":
    count = 0
    for py_file in BACKEND_DIR.rglob("*.py"):
        if fix_file(py_file):
            count += 1
    print(f"\nDone! Successfully updated {count} files.")