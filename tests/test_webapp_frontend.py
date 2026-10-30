"""Static checks for the web page (no browser needed).

Run from the VolleyIQ folder:
    .\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v

These catch the mistakes that silently break the page: JavaScript looking up
an element id that no longer exists, resources the Content-Security-Policy
would block, and missing accessibility basics.
"""

import re
import unittest
from html.parser import HTMLParser

from volleyiq.webapp.server import STATIC_DIR, STATIC_FILES


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.tags = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if "id" in attrs:
            self.ids.append(attrs["id"])


class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        cls.js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
        cls.css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")
        cls.page = PageParser()
        cls.page.feed(cls.html)

    def test_every_id_used_by_the_script_exists_once(self):
        used = set(re.findall(r'\$\("([\w-]+)"\)', self.js))
        views = re.search(r"const VIEWS = \[([^\]]+)\]", self.js).group(1)
        used |= {f"view-{name}" for name in re.findall(r'"(\w+)"', views)}
        sections = re.search(r"const SECTIONS = \[([^\]]+)\]", self.js).group(1)
        used |= set(re.findall(r'"(\w+)"', sections))
        missing = sorted(used - set(self.page.ids))
        self.assertEqual(missing, [], "app.js looks up ids that index.html doesn't have")
        duplicates = sorted({i for i in self.page.ids if self.page.ids.count(i) > 1})
        self.assertEqual(duplicates, [])

    def test_only_local_scripts_and_styles(self):
        for tag, attrs in self.page.tags:
            for name in ("src", "href"):
                value = attrs.get(name) or ""
                if tag in ("script", "link") and value and not value.startswith("data:"):
                    self.assertIn(value, STATIC_FILES, f"<{tag} {name}={value!r}> is not a served file")
        self.assertNotRegex(self.html + self.css, r"https?://(?!www\.w3\.org/2000/svg)")

    def test_no_inline_handlers_or_scripts(self):
        # The Content-Security-Policy blocks inline JavaScript.
        for tag, attrs in self.page.tags:
            self.assertFalse([a for a in attrs if a.startswith("on")], f"inline handler on <{tag}>")
            if tag == "script":
                self.assertIn("src", attrs)

    def test_accessibility_basics(self):
        self.assertIn('lang="en"', self.html)
        self.assertIn("prefers-reduced-motion", self.css)
        self.assertIn(":focus-visible", self.css)
        self.assertIn('class="skip-link"', self.html)
        timeline = next(attrs for tag, attrs in self.page.tags if attrs.get("id") == "timeline")
        self.assertEqual((timeline["role"], timeline["tabindex"]), ("slider", "0"))
        for tag, attrs in self.page.tags:
            if tag == "button":  # an explicit type, so no button submits a form by accident
                self.assertIn(attrs.get("type"), ("button", "submit"))

    def test_deleting_always_asks_first(self):
        for control in ("delete-results", "delete-failed", "cancel-job", "clear-history", "confirm-dialog"):
            self.assertIn(control, self.page.ids)
        self.assertIn('method: "DELETE"', self.js)
        # Every DELETE request sits in a function that awaits the confirmation dialog first.
        for function in re.findall(r"async function (\w+)\([^)]*\) \{(.*?)\n\}", self.js, re.S):
            name, body = function
            if 'method: "DELETE"' in body:
                self.assertLess(body.index("confirmAction("), body.index('method: "DELETE"'), name)
        # The result video stops streaming before its file is deleted (Windows can't delete open files).
        self.assertIn("releasePlayer()", self.js)

    def test_preview_is_allowed_by_the_content_security_policy(self):
        from volleyiq.webapp.server import SECURITY_HEADERS

        self.assertIn("media-src 'self' blob:", SECURITY_HEADERS["Content-Security-Policy"])
        self.assertIn("URL.createObjectURL", self.js)
        self.assertIn("URL.revokeObjectURL", self.js)

    def test_upload_can_be_cancelled_and_not_duplicated(self):
        self.assertIn("uploadRequest.abort()", self.js)
        self.assertRegex(self.js, r"if \(!selectedFile \|\| uploadRequest\) return;")

    def test_honest_scope_is_stated(self):
        text = re.sub(r"\s+", " ", self.html)
        self.assertIn("Not yet", text)
        self.assertIn("Tactical statistics and measured accuracy", text)
        self.assertIn("not hosted online", text)


if __name__ == "__main__":
    unittest.main()
