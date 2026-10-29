"""Static checks for the public website in docs/ (served by GitHub Pages).

Run from the VolleyIQ folder:
    .\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v

The website talks to the Hugging Face Space in space/app.py, so these checks
also make sure both sides agree on limits and file types.
"""

import ast
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

from volleyiq.webapp.pipeline import ALLOWED_EXTENSIONS

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CLIENT_URL = "https://cdn.jsdelivr.net/npm/@gradio/client@2.7.2/dist/browser.js"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.tags = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if "id" in attrs:
            self.ids.append(attrs["id"])


def space_constants():
    """MAX_UPLOAD_MB and MAX_DURATION_S from space/app.py, read without importing it (no Gradio needed)."""
    tree = ast.parse((ROOT / "space" / "app.py").read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant):
            values[node.targets[0].id] = node.value.value
    return values


class DocsSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (DOCS / "index.html").read_text(encoding="utf-8")
        cls.js = (DOCS / "app.js").read_text(encoding="utf-8")
        cls.config = (DOCS / "config.js").read_text(encoding="utf-8")
        cls.page = PageParser()
        cls.page.feed(cls.html)

    def test_every_id_used_by_the_script_exists_once(self):
        used = set(re.findall(r'\$\("([\w-]+)"\)', self.js))
        steps = re.search(r"const STEPS = \[([^\]]+)\]", self.js).group(1)
        used |= {f"step-{name}" for name in re.findall(r'"(\w+)"', steps)}
        self.assertEqual(sorted(used - set(self.page.ids)), [])
        self.assertEqual(sorted({i for i in self.page.ids if self.page.ids.count(i) > 1}), [])

    def test_limits_match_the_space(self):
        space = space_constants()
        self.assertIn(f"MAX_UPLOAD_MB = {space['MAX_UPLOAD_MB']};", self.config)
        self.assertIn(f"MAX_DURATION_S = {space['MAX_DURATION_S']};", self.config)
        listed = re.findall(r'"(\.\w+)"', re.search(r"ALLOWED_EXTENSIONS = \[([^\]]+)\]", self.config).group(1))
        self.assertEqual(listed, list(ALLOWED_EXTENSIONS))

    def test_gradio_client_is_pinned_and_integrity_checked(self):
        self.assertIn(f'from "{CLIENT_URL}"', self.js)
        preload = [a for t, a in self.page.tags if t == "link" and a.get("rel") == "modulepreload"]
        self.assertEqual(len(preload), 1)
        self.assertEqual(preload[0]["href"], CLIENT_URL)
        self.assertTrue(preload[0]["integrity"].startswith("sha384-"))
        self.assertEqual(preload[0]["crossorigin"], "anonymous")

    def test_only_expected_scripts(self):
        scripts = [a.get("src") for t, a in self.page.tags if t == "script"]
        self.assertEqual(scripts, ["app.js"])
        for tag, attrs in self.page.tags:
            self.assertFalse([a for a in attrs if a.startswith("on")], f"inline handler on <{tag}>")

    def test_no_secrets_and_backend_not_faked(self):
        text = self.html + self.js + self.config
        self.assertNotRegex(text, r"hf_[A-Za-z0-9]{30,}|(ghp|gho|ghs)_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}")
        self.assertNotIn("hf_token", text)
        # The backend is configured in one place; the localhost override only works on a local page.
        self.assertRegex(self.config, r'export const BACKEND = "[^"]*";')
        self.assertIn('["localhost", "127.0.0.1"].includes(location.hostname)', self.js)

    def test_privacy_notice_and_honest_scope(self):
        text = re.sub(r"\s+", " ", self.html)
        self.assertIn("Privacy.", text)
        self.assertIn("It is never shown to other visitors", text)
        self.assertIn("within about an hour", text)
        self.assertIn("Player tracking or identification", text)  # listed under "Not yet"
        self.assertIn("isn't live yet", text)  # the message shown while no backend is configured


if __name__ == "__main__":
    unittest.main()
