"""Assemble the files for the Hugging Face Space into one folder.

Usage:
    python space/build_space.py [OUTPUT_DIR]      (default: dist/space)

The output contains only what the Space needs: app.py, README.md (with the
Space settings), requirements.txt and the volleyiq Python package (without the
local app's web page and launchers). Upload the folder's contents to the Space,
for example with "Files > Add file > Upload files" on huggingface.co, or push it
to the Space's git repository. Model weights are not included: the Space
downloads the pinned model from Hugging Face when it starts.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE_FILES = ("app.py", "README.md", "requirements.txt")
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "static", "launcher.py")


def build(output_dir=ROOT / "dist" / "space"):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not (output_dir / "app.py").is_file():
            raise SystemExit(f"Refusing to replace {output_dir}: it doesn't look like a previous Space build.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for name in SPACE_FILES:
        shutil.copy2(ROOT / "space" / name, output_dir / name)
    shutil.copytree(ROOT / "volleyiq", output_dir / "volleyiq", ignore=SKIP)
    return sorted(str(p.relative_to(output_dir)).replace("\\", "/") for p in output_dir.rglob("*") if p.is_file())


if __name__ == "__main__":
    files = build(*sys.argv[1:2])
    print("\n".join(files))
    print(f"{len(files)} files")
