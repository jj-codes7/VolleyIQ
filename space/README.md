---
title: VolleyIQ
emoji: 🏐
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.10"
app_file: app.py
pinned: false
short_description: Volleyball ball detection and tracking for your videos
models:
  - grdesignbuild/volleyball-person-ball-detector
---

# VolleyIQ

Upload a volleyball clip (up to 60 seconds and 100 MB). VolleyIQ samples about
6 frames per second, detects the ball with the pretrained
[grdesignbuild/volleyball-person-ball-detector](https://huggingface.co/grdesignbuild/volleyball-person-ball-detector)
model (Apache-2.0) on a ZeroGPU GPU, tracks it with transparent rules, and
returns an annotated video, statistics and a per-frame tracking CSV.

It tracks the ball only. It does not track players, detect rallies or serves,
or produce tactical statistics, and its per-frame statuses are rules, not a
measured accuracy.

Uploaded videos are processed in this Space and are not kept: working files
are deleted when an analysis ends, results within about an hour.

Source code and the local app: https://github.com/jj-codes7/VolleyIQ
Website: https://jj-codes7.github.io/VolleyIQ/
