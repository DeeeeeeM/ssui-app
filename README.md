# NMI Media Toolkit

## Overview

`NMI Media Toolkit` is a Gradio-based user interface for speech, translation, and YouTube content processing. It combines AI-powered transcription, subtitle generation, and YouTube utilities to make audio/video handling easier.

Key capabilities:
- Speech-to-text transcription using OpenAI Whisper / stable-ts
- Local single and multiple file transcription
- YouTube video and playlist transcription support
- Subtitle translation and downloadable `.srt` files
- YouTube playlist extraction into CSV
- YouTube subtitle downloader
- YouTube tag checker for single videos and playlists

## Features

### Speech-to-text
- Select input mode for local single file, local multiple files, YouTube single video, or YouTube playlist processing
- Upload audio or video files or paste YouTube URLs
- Choose source language from supported Whisper languages
- Select model type: `faster whisper` or `whisper`
- Choose model size, including `large-v3`, `large-v2`, `large`, `medium`, `small`, `base`, `tiny`
- Download generated `.srt` file or ZIP package for multiple files
- Translate generated subtitles to another language
- View transcript directly in the UI
- Optional audio preprocessing and advanced segmentation controls

### YouTube utilities
- Playlist extractor: convert a YouTube playlist into a CSV with title, video ID, and URL
- SRT downloader: fetch English subtitles for one or more YouTube videos
- Tag checker: verify whether a specific metadata tag exists on a video
- Playlist tag checker: verify a tag across all videos in a playlist and export results to CSV

## Installation

1. Clone the repository:

```bash
git clone https://github.com/DeeeeeeM/ssui-app.git
cd ssui-app
```

2. Create a Python virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install system dependencies:
- `ffmpeg` must be available on your PATH
- optionally install `yt-dlp` globally for improved YouTube subtitle downloads:

```bash
gpip install yt-dlp
```

5. Optional extras:
- `demucs` for vocal separation support
- GPU-enabled PyTorch if using CUDA with stable-ts/faster-whisper

## Usage

Run the app:

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal.

### Primary workflow

1. Open the `Speech to Text` tab
2. Select the input mode: local single, local multiple, YouTube single, or YouTube playlist
3. Upload files or paste YouTube URLs
4. Select source language and model settings
5. Click `PROCESS`
6. Download transcript and subtitle file(s) from the results section

### YouTube workflows

- `Youtube playlist extractor`: paste a playlist URL and download a CSV of video title/ID/URL
- `SRT Downloader`: paste one or more video URLs and download English subtitle files
- `Tag Checker`: paste a video URL and tag to verify whether the tag exists
- `Playlist Tag Checker`: paste a playlist URL and a tag, then download a CSV report
