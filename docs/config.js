// Where the website sends videos for analysis: the VolleyIQ Hugging Face Space.
// Use the Space id ("owner/space-name") or its URL ("https://owner-space-name.hf.space").
// Leave it empty until the Space is deployed: the page then says the online
// analysis isn't available yet instead of trying to connect.
export const BACKEND = "";

// Limits shown to visitors. They must match space/app.py (the server enforces them).
export const MAX_UPLOAD_MB = 100;
export const MAX_DURATION_S = 60;
export const ALLOWED_EXTENSIONS = [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"];
