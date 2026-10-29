"""Start VolleyIQ without a console window. It opens in your browser.

Prefer VolleyIQ.vbs: on some computers Windows asks which app should open
.pyw files instead of running them. Both run the same launcher.

If something goes wrong, a message box explains it. To stop VolleyIQ, use
"Stop VolleyIQ" in the web page. VolleyIQ.bat does the same with a console
window, if you prefer to see the output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from volleyiq.launcher import run  # noqa: E402  (needs the path above)

sys.exit(run())
