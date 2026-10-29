# VolleyIQ

Volleyball video analytics. The current prototype reads a local volleyball
video, saves selected frames as images, and lets a person mark the ball by
hand to build ground-truth labels. Every ball position in the annotation CSV
comes from a human click.

Separately, `volleyiq.detect_ball` runs a pretrained detector on extracted
frames. Its results are automatic, are kept in `outputs\detections\`, and
never change the manual annotation CSV.

**The easiest way in is the web app:** double-click `VolleyIQ.vbs`, upload a
video, and get it back with the ball detected and tracked. See
[Web app](#web-app).

## Project layout

```
VolleyIQ/
├── .venv/                  Python virtual environment (not in Git)
├── data/
│   ├── videos/             put your own volleyball clips here (not in Git)
│   ├── frames/             extracted frames are saved here (not in Git)
│   └── annotations/        ball-position CSVs (created when you annotate)
├── app_data/               web app uploads and results (not in Git, auto-deleted)
├── models/                 downloaded model weights (not in Git)
├── outputs/                generated results (not in Git)
│   ├── detections/         automatic detection results
│   ├── tracking/           rule-based ball tracking results
│   └── evaluation/         strategy comparisons against the hand labels
├── volleyiq/
│   ├── __init__.py
│   ├── extract_frames.py   saves every Nth frame of a video
│   ├── make_test_video.py  draws a synthetic test video (not real footage)
│   ├── annotate_ball.py    click on the ball to label frames by hand
│   ├── visualize_trajectory.py  draws the hand-labelled ball path
│   ├── detect_ball.py      automatic ball detection with a pretrained model
│   ├── track_ball.py       picks one ball per frame from those detections
│   ├── evaluate_tracking.py  compares tracking strategies with the hand labels
│   └── webapp/             the local web app (server, jobs, pipeline, web page)
├── tests/                  automated checks (no clicking needed)
├── requirements.txt        exact package versions used
├── VolleyIQ.vbs            double-click to start the web app (no console window)
├── VolleyIQ.bat            the same, with a console window
├── VolleyIQ.pyw            the same as the .vbs, where .pyw files open with Python
└── README.md
```

## Setup

The project uses Python 3.10 and the virtual environment in `.venv`.
All commands below are for PowerShell and should be run from the
`VolleyIQ` folder.

If `.venv` already exists and works, you do not need to install anything.

To recreate the environment from scratch:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pulls the CUDA 13.0 builds of PyTorch from the official
PyTorch index. PyTorch, Transformers, Hugging Face Hub and safetensors are
only needed for automatic detection, not for frame extraction or manual
annotation.

Using `.\.venv\Scripts\python.exe` directly means you never have to
"activate" the environment, and nothing is installed globally.

## Web app

A local web page where you upload a volleyball video and get it back with
the ball detected and tracked. It uses the same detector, model and tracker
as the command-line tools below, and runs entirely on this computer.

This is a local app: it runs on this computer at <http://127.0.0.1:8765/>
and is not hosted on the internet.

### Start it

**Double-click `VolleyIQ.vbs`** in the `VolleyIQ` folder. No console window
appears: VolleyIQ starts in the background (the first start after turning
on the computer can take 20 to 30 seconds while PyTorch loads; later ones
take a few seconds), then your browser opens. If it was already running,
the browser simply opens. If it can't start, a message box shows why.

**To stop it**, click **Stop VolleyIQ** at the top right of the page. If an
analysis is still running, the page warns you first: it would be
interrupted.

Prefer to see the output? **Double-click `VolleyIQ.bat`** instead. It does
the same with a console window; close the window (or press Ctrl+C in it)
to stop. Use this if `VolleyIQ.vbs` reports a problem.

You can make a desktop shortcut to either file (right-click, *Send to*,
*Desktop (create shortcut)*); the launchers find the project from their own
location, so shortcuts work. From PowerShell:

```powershell
.\.venv\Scripts\python.exe -m volleyiq.webapp
```

Options: `--port 8765`, `--no-browser`, `--max-upload-mb 500`, `--log-file PATH`.
No extra packages are needed: the server uses Python's built-in
`http.server`, and the launcher only the standard library.

How `VolleyIQ.vbs` works: Windows runs `.vbs` files with its built-in
`WScript.exe`, so no file association has to be set up. The script works
out the project folder from its own location and runs
`.venv\Scripts\pythonw.exe volleyiq\launcher.py` with no window. The
launcher starts `pythonw -m volleyiq.webapp` in the background, with all
output written to `app_data\logs\server.log` (started afresh above 5 MB),
waits for `/api/health` to answer, and opens the browser. If an older
VolleyIQ (started before this version) is already running, it opens that
one and tells you to restart it to get the new version. If Python itself
can't run, `VolleyIQ.vbs` shows its own message.

`VolleyIQ.pyw` runs the same launcher, but only on computers where
double-clicking `.pyw` files runs Python. On many, Windows instead asks
which app to open it with, because no default app was ever chosen for
`.pyw` files; use `VolleyIQ.vbs` there.

### Use it

1. Click **Analyze a match**, then drag a video onto the upload area (or
   click *browse*, or press Enter on it), and click **Analyze a match**
   again. Accepted: MP4, MOV, M4V, AVI, MKV and WEBM, up to 500 MB and
   5 minutes. The server checks the file really is a readable video before
   accepting it.
2. The page shows the job's state: **Queued**, then **Processing** with a
   step tracker (extract frames, load AI model, detect ball with a frame
   counter, track ball, render video) and the time so far. You can close
   the tab; the job keeps running.
3. When it's **Completed** you get a results dashboard:
   - the analysed video, with normal playback controls;
   - a **tracking timeline**: one coloured mark per analysed frame (green
     good, orange uncertain, grey lost, red bad). Click it, drag along it,
     or focus it and use the arrow keys (Page Up/Down for 10 frames,
     Home/End) to jump the video to that frame. Below it, the current
     frame's status, ball position, model score and the tracker's reason;
   - statistics, all from this analysis: share of frames with a ball,
     counts per status, video length, size and frame rate, sampling, raw
     detector boxes, static look-alikes ignored, processing time per step,
     and GPU or CPU;
   - downloads for the video and the tracking CSV.
4. If it **Failed**, the page says why (for example "could not be opened as
   a video" or "too long") and lets you try another file.

**Recent analyses** on the start page lists each job with its status and,
when finished, the share of frames with a ball. Each has its own link
(`#job/<id>`) you can reopen until it expires.

The page states what this version does and doesn't do: it detects and
tracks the ball, but does not yet track players, detect rallies or
serves, or produce tactical statistics, and its statuses are not a
measured accuracy. It works with the keyboard, shows focus clearly, and
turns off animation when the system asks for reduced motion.

**What the analysed video shows.** About 6 frames per second are analysed
(every 5th frame of 30 FPS video), and the video plays those frames at
real-time speed, so it looks choppier than the original. Each frame shows
the tracking status (good, uncertain, lost, bad), the selected ball and its
recent trail, exactly like `track_ball`'s preview. Positions are never
filled in between analysed frames. The statuses mean the same as in
[Track the ball across frames](#track-the-ball-across-frames).

### How it works

```
browser ──upload──▶ server (127.0.0.1) ──▶ job queue ──▶ one worker thread
   ▲                     │                                   │
   └──poll status────────┘   extract_frames ▶ detect_ball.run_detection
   └──play video─────────┘   ▶ track_ball.track (baseline) ▶ render video
```

- `volleyiq/webapp/server.py`: the HTTP server and API.
- `volleyiq/webapp/jobs.py`: the job queue and storage.
- `volleyiq/webapp/pipeline.py`: the analysis, built from the existing
  modules. The model is loaded once, on the first job, and reused.
- `volleyiq/webapp/static/`: the web page (plain HTML, CSS and JavaScript,
  no external files).

| Endpoint | Purpose |
|---|---|
| `POST /api/jobs?filename=NAME` | Upload the video (raw body, `application/octet-stream`); returns the new job |
| `GET /api/jobs/<id>` | Job status: `queued`, `processing` (with `stage` and progress), `completed` (with `stats`) or `failed` (with `error`) |
| `GET /api/jobs/<id>/video` | The analysed video (only when completed; supports seeking) |
| `GET /api/jobs/<id>/track.csv` | Per-frame tracking result (only when completed) |
| `GET /api/jobs/<id>/tracking` | The same per-frame result as JSON, for the timeline |
| `GET /api/jobs`, `GET /api/config` | Recent jobs; upload limits and pipeline settings |
| `GET /api/health` | `{"app": "VolleyIQ"}`, so the launcher can tell VolleyIQ is running |
| `POST /api/shutdown` | Stop VolleyIQ (needs the header `X-VolleyIQ-Action: shutdown`) |

- `volleyiq/launcher.py`, started by `VolleyIQ.vbs` (or `VolleyIQ.pyw`): the no-console launcher.

Settings come from the existing tools: tile 864, threshold 0.4, the
baseline tracker, and `--max-displacement` scaled from 120 px at 1920 px
wide to the video's width. The analysed video is H.264 MP4, written by
OpenCV through Windows' built-in encoder; if that isn't available it falls
back to VP8 WebM. Either plays in Edge, Chrome and Firefox.

### Where files go, and how long they stay

Everything is under `app_data\jobs\<job id>\`, with a random id, never the
uploaded file's name. Nothing else in the project is read or written:
not `data\`, `outputs\`, the annotation CSVs or the model cache.

- The uploaded video, extracted frames and detection files are **deleted as
  soon as the job finishes**, whether it succeeded or failed.
- The analysed video, `track.csv` and `job.json` are kept for **24 hours**,
  then deleted automatically (checked when the app starts and after every
  job). At most the 20 newest finished jobs are kept.
- To remove everything at once, stop the app and delete the `app_data`
  folder.

### Safety

- Listens on `127.0.0.1` only, so other computers can't connect.
- Requests from other websites (and DNS-rebinding tricks) are refused by
  checking the `Host` and `Origin` headers.
- The upload's name is only shown on screen; files are saved under fixed
  names. Job ids must be 16 hex characters, and only `index.html`,
  `app.css` and `app.js` are served as files, so no other path on disk can
  be reached through the app.
- Uploads are checked for file type, size (refused from the headers before
  reading, if too large) and that OpenCV can actually decode them.
  Interrupted uploads are discarded.
- A job is only marked completed after its output video has been read back.
  Jobs running when the app was closed are marked failed ("Interrupted") on
  the next start.
- The stop request needs a custom header, which another website can't send
  without the browser first asking VolleyIQ (and VolleyIQ never agrees), so
  only the VolleyIQ page itself can stop the app.

### Limitations

- One person on one computer. Videos are processed one at a time; up to 3
  can wait.
- Processing takes time: on the RTX 3050 Laptop GPU, a 65-second 1080p clip
  took about 2 minutes (see below). Without CUDA it falls back to the CPU,
  which is much slower.
- The analysed video is large (81 MB for that clip), because Windows'
  H.264 encoder uses a high bitrate and OpenCV can't change it. It is
  deleted with the job after 24 hours.
- Only about 6 frames per second are analysed, so the result video is
  choppier than the original, and very fast moments between analysed
  frames are not seen.
- Detection and tracking have all the limitations described below: false
  positives (arms, heads, round objects), misses on motion-blurred balls,
  and the tracker's `good` status is not a measured accuracy. The model was
  trained on a different camera setup than most phone footage.
- Clips over 5 minutes are refused; trim them first.

**Checked result.** Uploading `real_volleyball.mp4` (42.7 MB, 65 s,
1920 x 1080) through the app: 391 frames analysed on CUDA in 126 s
(extract 6 s, load model 7 s, detect 102 s, track 4 s, render 7 s), giving
an H.264 MP4 that played and seeked in Edge. Its tracking matched the
command-line baseline in every frame (295 good, 11 uncertain, 81 lost,
4 bad).

### Troubleshooting

| Problem | What to do |
|---|---|
| The `.bat` window or a message box says it can't find `.venv` | Set up the environment (see [Setup](#setup)). |
| Double-clicking `VolleyIQ.pyw` asks "Select an app to open this .pyw file" | Windows has no default app chosen for `.pyw`. Double-click `VolleyIQ.vbs` instead (or `VolleyIQ.bat`). |
| Double-clicking `VolleyIQ.vbs` opens it in an editor, or says it can't run scripts | Your Windows opens `.vbs` files differently or has VBScript turned off (Microsoft plans to phase VBScript out of future Windows versions). Use `VolleyIQ.bat`. |
| A message box says VolleyIQ could not start | Read the log lines it shows (also in `app_data\logs\server.log`), or start `VolleyIQ.bat` to see the full output. |
| "An older version of VolleyIQ is already running" | Close the old console window (or press Ctrl+C in it), then start VolleyIQ again. |
| "Could not start on port 8765" | VolleyIQ is probably already running: open <http://127.0.0.1:8765/>. Or start it with `--port 8766`. |
| The page header says "Not reachable" | VolleyIQ has stopped. Start it again. |
| Job failed: "model could not be loaded from models/hf_cache" | The model files are missing. With internet access, run `detect_ball` once on a few frames (see below) to download them, then restart the app. |
| Job failed: "could not be opened as a video" | The file is damaged or uses a codec OpenCV can't read. Re-export it as an H.264 MP4. |
| The video doesn't play in the browser | Use *Download video* and open it in a video player, or try Edge or Chrome. |
| A job says "Interrupted" | The app was closed while it was running. Upload the video again. |
| Processing is very slow | Check the result's "on CUDA/CPU" line. CPU means PyTorch can't see the GPU. |

## Extract frames from a video

1. Copy a volleyball video you recorded (or have the right to use) into
   `data\videos\`, for example `data\videos\rally1.mp4`.
2. Run:

```powershell
.\.venv\Scripts\python.exe -m volleyiq.extract_frames data\videos\rally1.mp4 data\frames\rally1 --every 30
```

- `data\videos\rally1.mp4` is the video to read.
- `data\frames\rally1` is the folder to save frames in (created if missing).
- `--every 30` saves one frame out of every 30. At 30 FPS that is one
  frame per second. Leave it out to use the default of 30, or use
  `--every 5` for more frames.

The command prints the video's size, FPS, total frames and duration, then
saves images named after their original frame number:

```
data\frames\rally1\frame_000000.jpg
data\frames\rally1\frame_000030.jpg
data\frames\rally1\frame_000060.jpg
```

So `frame_000060.jpg` is frame 60 of the original video.

If the path is wrong or the file cannot be read as a video, the command
prints an `Error:` message and exits without saving anything.

## Make a synthetic test video (optional)

If you don't have footage yet, you can draw a simple test clip: a cartoon
court, a net, and a ball flying back and forth. Every frame is labelled
**"SYNTHETIC TEST VIDEO - not real footage"**. It is only for checking that
the tools work; it is not real match footage and says nothing about how well
any AI model would track a real ball.

```powershell
.\.venv\Scripts\python.exe -m volleyiq.make_test_video --output data\videos\synthetic_test.mp4 --duration 4
.\.venv\Scripts\python.exe -m volleyiq.extract_frames data\videos\synthetic_test.mp4 data\frames\synthetic_test --every 30
```

Options: `--output` (default `data\videos\synthetic_test.mp4`), `--duration`
in seconds (default 4), `--width`/`--height` (default 640 x 360) and `--fps`
(default 30).

## Mark the ball by hand

This is manual labelling: you click the ball, and the tool records where you
clicked. Nothing is detected automatically.

```powershell
.\.venv\Scripts\python.exe -m volleyiq.annotate_ball data\frames\rally1 data\annotations\rally1_ball.csv
```

- `data\frames\rally1` is a folder of frames from `extract_frames`.
- `data\annotations\rally1_ball.csv` is where the labels are saved (the
  folder is created if needed).

A window opens showing one frame, with an info panel above it and the
controls below it.

| Control | What it does |
|---|---|
| Left click | Mark the centre of the ball. Click again to correct it. |
| `D` or Right arrow | Next frame |
| `A` or Left arrow | Previous frame |
| `S` | Skip: the ball is not visible in this frame. Moves to the next frame. |
| `C` | Clear this frame's mark (back to "not annotated") |
| `Q` or `Esc` | Quit (closing the window also works) |

The marked point is shown as a red circle with a cross.

**Your work is saved after every click, skip or clear**, so quitting or
closing the window never loses anything. Running the same command again
reopens the CSV and jumps to the first frame you haven't done yet.

Large frames (for example 1920 x 1080) are shrunk to fit the window, but
coordinates are always saved in the original frame's pixels. Use
`--max-width` and `--max-height` to change the window limit (default
1280 x 720).

### CSV format

The CSV has one row per frame, in frame order:

```
frame_name,x,y,status
frame_000000.jpg,212,224,labelled
frame_000030.jpg,,,skipped
frame_000060.jpg,,,
```

- `x`, `y`: the ball centre in pixels, measured from the top-left corner of
  the frame. Blank if the frame was skipped or not done yet.
- `status`: `labelled`, `skipped` (ball not visible), or blank (not done yet).

If the output path points to a CSV that isn't a VolleyIQ annotation file,
the tool stops with an error instead of overwriting it.

## Draw the ball's path

This draws the positions **you clicked** in the annotation tool. It is not
automatic ball detection: if you didn't label a frame, nothing is drawn for
it. Every output is labelled "Manual annotations - not automatic detection".

**As an image**: the whole path drawn on one extracted frame.

```powershell
.\.venv\Scripts\python.exe -m volleyiq.visualize_trajectory image data\annotations\rally1_ball.csv data\frames\rally1 outputs\rally1_trajectory.png
```

By default the first labelled frame is used as the background. Pick a
different one with `--frame frame_000060.jpg`. The output can be `.png` or
`.jpg`.

**As a video**: the original video, with the path growing as it plays.

```powershell
.\.venv\Scripts\python.exe -m volleyiq.visualize_trajectory video data\annotations\rally1_ball.csv data\videos\rally1.mp4 outputs\rally1_trajectory.mp4
```

Each point appears on the exact video frame it was labelled on and stays
for the rest of the video. This works because `extract_frames` names every
image after its original frame number (`frame_000060.jpg` is video frame
60). Use the same video you extracted the frames from. If the CSV mentions a
frame the video doesn't have, the tool stops with an error.

**What the drawing means**

- Green dot: the first labelled point. Red dots: the other labelled points.
- Orange lines join points from consecutive annotated frames.
- If a frame in between was **skipped** (ball not visible) or **not
  annotated**, the line is broken there. The tool never guesses where the
  ball went.

With `--every 30`, consecutive points are a second apart, so the straight
lines between them are a rough sketch, not the real curved flight. Extract
with a smaller interval (for example `--every 5`) for a smoother path.

## Detect the ball automatically

This runs a pretrained AI model on extracted frames and draws a box around
each ball it finds. Unlike everything above, **these positions are not
human labels**: they are model guesses and can be wrong. This is per-frame
object detection only; it does not track the ball from frame to frame.

```powershell
.\.venv\Scripts\python.exe -m volleyiq.detect_ball data\frames\real_volleyball --limit 5 --manual-csv data\annotations\real_volleyball_ball.csv
```

Options:

| Option | What it does |
|---|---|
| `--frames NAME ...` | Only these frames, e.g. `--frames frame_000000.jpg frame_000035.jpg` |
| `--limit N` | At most N frames (the first N, in frame order) |
| `--threshold 0.4` | Minimum model confidence (0 to 1) for a box to be kept. Default 0.4. |
| `--tile 864` | Tile size in pixels; `0` runs the whole frame only. Default 864. |
| `--manual-csv CSV` | Draw your hand labels as well, for comparison. The CSV is only read. |
| `--keep-person` | Also keep `person` boxes (dropped by default) |
| `--out DIR` | Output folder. Default: `outputs\detections\<frames>_tile<N>_t<threshold>` |

The output folder must be new or empty: the tool stops with an error rather
than overwrite earlier results. It contains:

- `<frame>.auto.jpg`: the frame with magenta **AUTO** boxes and scores. With
  `--manual-csv`, your hand label is drawn as a cyan **MANUAL** cross. A key
  in the corner says which is which.
- `<frame>.auto.json`: that frame's detections (`label`, `score`, `box` as
  x1, y1, x2, y2 pixels, `centre`, `"source": "automatic"`), the model
  revision and settings, and, if a manual CSV was given, the hand label as
  `manual_reference` and the distance to the nearest AUTO ball.
- `ball_detections_auto.csv`: one row per detected ball
  (`frame_name,x,y,score,x1,y1,x2,y2,source`), where `x`, `y` is the box
  centre so it lines up with the manual CSV. Frames with no ball get a row
  with blank values. This file is **not** an annotation CSV; don't use it as
  ground truth.
- `run_info.json`: model, revision, device and settings for the run.

**Tiling.** The ball is small, so large frames are cut into overlapping
square tiles (25% overlap) plus one whole-frame pass, and duplicate boxes are
merged. A 1920 x 1080 frame becomes 7 passes at `--tile 864` and 9 at
`--tile 640`. The tiling and merging follow the model's own `predict.py`.

**Model download.** The first run downloads about 41 MB of weights from
Hugging Face into `models\hf_cache\` inside this project (git-ignored). The
model is pinned to one exact revision, so results don't change if the model
author publishes an update. After that, you can run offline:

```powershell
$env:HF_HUB_OFFLINE = "1"
```

On the RTX 3050 Laptop GPU, one 1080p frame takes about 0.2 to 0.3 s after
the first. Without CUDA it falls back to the CPU, which is much slower.

**Limitations**

- The model was trained on a different camera (high on the referee stand or
  behind the end line), not on footage like ours. Results here are a first
  check on 5 frames, not a measured accuracy.
- It finds things that look like a ball. On `real_volleyball`, players'
  hands were sometimes boxed as balls, and a small round object near the
  net was detected in most frames even though it never moves. Expect false
  positives, and more of them at lower thresholds and smaller tiles.
- Motion-blurred balls are the most common miss (from the model card).
- It reports every ball it sees, including spare balls; it does not know
  which one is in play.

**Model and licence.** [grdesignbuild/volleyball-person-ball-detector](https://huggingface.co/grdesignbuild/volleyball-person-ball-detector),
revision `92693c99a63841b1b121df86607bb4f839bbe107`: a fine-tune of
[D-FINE-S](https://huggingface.co/ustc-community/dfine-small-obj2coco)
detecting `person` and `ball`, released under the Apache-2.0 licence. The
tiling and merging code in `detect_ball.py` is adapted from that
repository's `predict.py` (Apache-2.0).

## Track the ball across frames

The detector often finds more than one "ball" in a frame, or a "ball" that
is really an arm or a wall decoration. `track_ball` reads the detector's
`<frame>.auto.json` files (it does not run the model again) and picks **at
most one** ball per frame using simple, explainable rules. When no candidate
fits, the frame gets no position: positions are never guessed or
interpolated.

```powershell
.\.venv\Scripts\python.exe -m volleyiq.track_ball outputs\detections\full_clip_tile864_t040 --max-displacement 120
```

| Option | What it does |
|---|---|
| `DETECTIONS_DIR` | Folder of `<frame>.auto.json` files from `detect_ball` |
| `--out DIR` | New or empty output folder. Default: `outputs\tracking\<detections>_maxd<N>` |
| `--max-displacement 120` | Largest believable ball move, in pixels, between two sampled frames. Default 120. |
| `--max-lost 2` | Frames without a ball before the track is dropped and restarted. Default 2. |
| `--min-chain 3` | Consecutive linked frames needed for `good`. Default 3. |
| `--min-good-score 0.5` | Lowest model score allowed for `good`. Default 0.5. |
| `--static-fraction 0.2` | Ignore any spot detected in at least this share of frames; `0` turns it off. Default 0.2. |
| `--frames-dir DIR` | Frames for the preview. Default: the folder recorded in the detections' `run_info.json`. |
| `--fps 6` | Preview video speed. 6 matches frames extracted with `--every 5` from ~30 FPS video. |
| `--no-video` | Skip the preview video |
| `--strategy baseline` | `baseline` (default) or `motion` (experimental, see below) |

The default of 120 px comes from the hand labels on `real_volleyball`: the
fastest ball moved 88 px between frames extracted 5 video frames apart.
Change it if you extract frames at a different interval or the ball is
closer to the camera.

**The rules**, applied frame by frame in time order:

1. **Static objects are ignored.** A spot where a "ball" is detected in at
   least 20% of all frames is treated as background. On `real_volleyball`
   this removes a red disc on the wall that was detected in 310 of 391
   frames.
2. **No active track:** the highest-scoring remaining candidate starts one.
3. **Active track:** only candidates within the maximum displacement of the
   last selected position count (twice that after one missing frame, and so
   on). Among them, the one with the lowest
   `distance / allowed + (1 - score)` is chosen, so closeness and confidence
   both matter.
4. After `--max-lost` frames in a row with no selection, the track is
   dropped and rule 2 starts a new one.

**How to read the statuses**

| Status | Position? | Meaning |
|---|---|---|
| `good` | yes | Part of an unbroken chain of at least 3 consecutive frames, each step within the maximum displacement; the only candidate in range; score at least 0.5. **A high score alone never makes a frame good.** |
| `uncertain` | yes | A ball was selected, but at least one `good` condition fails. The `reason` column says which: a short chain (a new or interrupted track), other candidates in range, or a low score. Check these before relying on them. |
| `lost` | no | The detector found no ball, or only the static object. |
| `bad` | no | The detector found ball candidate(s), but all were too far from the track to be the same ball. Usually false positives; sometimes a real ball moving faster than the limit. |

**Outputs**, in the output folder (which must be new or empty; existing
results are never overwritten):

- `ball_track.csv`: one row per frame with `frame_name`, `frame_number`,
  `status`, `reason`, `x`, `y` (blank when there's no position), `score`, and
  diagnostics: `track_id`, `chain_length`, `n_candidates` (ball boxes from the
  detector), `n_static`, `n_in_range`, `displacement_px` (from the last
  selected position), `steps_since_selected`, `allowed_px` and
  `nearest_rejected_px` (for `bad` frames). This is **not** an annotation
  CSV; don't use it as ground truth.
- `track_preview.mp4`: each frame with the status and reason at the top
  left, a ring on the selected ball, and the recent trail (lines only join
  consecutive frames of one chain). Green = good, orange = uncertain,
  grey = lost, red = bad. Raw detector boxes are not drawn; use
  `detect_ball`'s images for those.
- `track_info.json`: settings, static spots found, status counts and, for
  frames that had a manual label when `detect_ball` ran with `--manual-csv`,
  the distance between the tracked and hand-labelled positions.

**Results on `real_volleyball`** (391 frames, default settings): 295 good,
11 uncertain, 81 lost, 4 bad. All 20 hand-labelled frames were `good` and
within 13 px of the hand label. In a visual spot check, the sampled `good`
frames all showed the real ball; some `uncertain` frames were hands or arms
and others were the real ball at the start of a short chain.

**Limitations.** This is a simple baseline, not a solved problem:

- It removes many false positives but not all. A false positive that moves
  smoothly for 3 or more frames (for example a hand following the ball) can
  still be marked `good`, and if the track starts on a false positive it
  can follow it.
- Real balls are sometimes `uncertain` (short chains, two candidates in
  range) or `bad` (moved faster than `--max-displacement`), so some correct
  positions get no `good` rating.
- `lost` frames depend entirely on the detector: if it misses the ball,
  tracking can't recover it. Positions in gaps are never filled in.
- The static rule can hide a real ball that stays in one place for a long
  time (for example a ball held before a serve), or that passes within
  12 px of the static spot. On short clips (few frames), lower or disable
  it with `--static-fraction`.
- It follows one ball. It can't tell which of two real balls is in play.
- `good` was checked against only 20 hand labels and a visual spot check on
  one clip. It is not a measured accuracy.

### Motion-aware strategy (experimental)

```powershell
.\.venv\Scripts\python.exe -m volleyiq.track_ball outputs\detections\full_clip_tile864_t040 --strategy motion
```

The baseline ranks the candidates in range by distance from the **last**
position. The `motion` strategy ranks them by distance from a **predicted**
position instead. Everything else is the same: the static-object rule, the
maximum-displacement gate, track resets and the status rules. The baseline
stays the default, and its output is unchanged.

- **Prediction.** Constant velocity from the last two *reliable* positions
  of the current track, measured in pixels per video frame (using the
  numbers in the frame names), so it still works across missing frames. If
  there aren't two reliable positions, or they are more than
  `--max-lost` + 1 sampled steps apart, there is no prediction and the last
  position is used, exactly like the baseline.
- **Ranking.** `distance to prediction / allowed + (1 - score)`, lowest wins.
- **Ambiguity.** With several candidates in range, the frame counts as
  ambiguous (so `uncertain`) only when the runner-up's cost is within 0.25
  of the winner's. The baseline treats any two candidates in range as
  ambiguous.
- **What counts as reliable**, and so can feed the velocity: a selection
  that is not ambiguous, has a score of at least `--min-good-score` and, when
  it starts a track, is the only candidate. Other selections still move the
  gate, but never the velocity, so one bad pick can't bend the prediction.
  `lost` and `bad` frames add nothing.
- **Resets and recovery.** After `--max-lost` frames without a selection,
  the track and its velocity are dropped. The next candidate starts a new
  track with no prediction; prediction resumes once the new track has two
  reliable positions.
- **The gate stays around the last position, not the prediction.** On
  `real_volleyball` the ball's distance from the prediction was usually
  small (median 3.4 px) but reached 175 px when the ball was hit and
  changed direction. A gate around the prediction would have rejected those
  real balls.

The motion strategy's CSV has extra columns: `predicted_x`, `predicted_y`,
`prediction_error_px`, `velocity_x`, `velocity_y` (pixels per video frame),
`cost_margin` and `reliable`. Its preview also marks the predicted position
with a small white x. Its default output folder is
`outputs\tracking\<detections>_motion_maxd<N>`.

## Compare strategies with the hand labels

```powershell
.\.venv\Scripts\python.exe -m volleyiq.evaluate_tracking outputs\detections\full_clip_tile864_t040 data\annotations\real_volleyball_ball.csv
```

This runs both strategies on the same detections (no video) and scores them
against your hand labels. **Only rows marked `labelled` with x and y are
ground truth.** Frames marked `skipped` are reported but not scored, because
you may skip a frame where the ball is only partly visible. Frames you never
annotated are ignored. The manual CSV is only read.

For each strategy it reports: labelled frames, **matched** (the tracker gave
a position, any status), **missed** (no position), mean, median and maximum
pixel error over matched frames, how many are within 5, 10 and 15 px, the
statuses on labelled frames, and how many positions it gave on skipped
frames.

Output (default `outputs\evaluation\<detections>_maxd<N>`, must be new or
empty): `baseline_track.csv`, `motion_track.csv`, `labelled_frames.csv` (each
labelled frame with every strategy's answer and error) and
`comparison.json` (settings, metrics, prediction diagnostics and every frame
where the strategies disagree). `--out` and `--max-displacement` work as in
`track_ball`.

**Results on `real_volleyball`** (391 frames, 20 labelled, defaults):

| Strategy | Labelled | Matched | Missed | Mean px | Median px | ≤5 px | ≤10 px | ≤15 px | Good |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 20 | 20 | 0 | 4.4 | 3.6 | 15 | 18 | 20 | 20 |
| motion | 20 | 20 | 0 | 4.4 | 3.6 | 15 | 18 | 20 | 20 |

**The two strategies gave identical results on every one of the 391
frames.** That's because, on this clip, there was never more than one
candidate in range while a track was running (once the static red disc is
ignored), and ranking is the only thing the motion strategy changes. Both
gave positions on 6 of the 22 skipped frames (3 `good`, 3 `uncertain`); in an
earlier visual check, the `good` ones were a real ball, half-hidden or cut
off at the top of the picture.

**Limitations of this comparison**

- 20 labelled frames from one clip, most of them in easy, steady flight.
  This can't show a general accuracy, and can't show a difference between
  the strategies, because they never disagreed here.
- The motion strategy's advantage only appears when two candidates are in
  range at once (for example a hand near the ball). The unit tests cover
  that case with made-up data; this clip doesn't contain it.
- Constant velocity ignores gravity and spin, so the prediction is weaker
  on curved, slow arcs and wrong at every hit or bounce.
- Both strategies share every other limitation listed for tracking above.
  To learn more, label more frames, especially ones with several
  candidates or fast play.

## Run the automated checks

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The checks cover frame extraction, the synthetic video, the annotation
logic (saving, loading, skipping, navigation, error messages), the
trajectory drawing (line breaks at skipped frames, image and video output,
frame-number matching), the automatic detection logic (tiling, merging
duplicate boxes, output files, refusing to overwrite) and the tracking rules
(missing detections, competing candidates, large jumps, recovery after lost
frames, static objects, output files), the motion-aware strategy (steady
movement, direction changes, gaps, recovery, competing candidates, sudden
jumps, keeping bad picks out of the velocity), the evaluation (only
labelled frames scored, error statistics) and the web app (upload checks,
job status changes, failed and interrupted jobs, video download with
seeking, blocked paths and foreign websites, the timeline data, stopping
the app, and the full pipeline on a synthetic video), the web page (every
element the script needs exists, nothing external or inline is loaded,
keyboard and reduced-motion basics) and the no-console launcher (detecting
a running or older VolleyIQ, startup failures shown with the log). The detection and tracking checks
use made-up detections instead of the model, so they don't download any
weights or need a network. They don't
open any windows or need anyone to click, and they write only to a
temporary folder that is deleted afterwards. `-B` stops Python from creating
`__pycache__` folders.
