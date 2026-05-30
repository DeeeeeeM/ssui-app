from modules.utils import normalize_file_path

import os
import shutil
import subprocess
import glob
import tempfile
from yt_dlp import YoutubeDL

def download_srt(video_urls, cookies_path=None):
    try:
        if not video_urls:
            return None, "No URL provided"

        if isinstance(video_urls, (list, tuple)):
            urls = [u.strip() for u in video_urls if u and u.strip()]
        else:
            parts = []
            for line in str(video_urls).splitlines():
                for part in line.split(','):
                    parts.append(part.strip())
            urls = [p for p in parts if p]

        if not urls:
            return None, "No URL provided"

        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        output_template = os.path.join(downloads_dir, "%(id)s.%(ext)s")

        errors = []
        cookies_path = normalize_file_path(cookies_path)
        try:
            if shutil.which("yt-dlp"):
                for url in urls:
                    if not url:
                        continue
                    cmd = [
                        "yt-dlp",
                        "--write-subs",
                        "--write-auto-subs",
                        "--sub-lang", "en-US",
                        "--skip-download",
                        "--convert-subs", "srt",
                        "-o", output_template,
                        # pass cookies if provided
                        url
                    ]
                    if cookies_path:
                        cmd.extend(["--cookies", cookies_path])
                    try:
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                        print(result.stdout)
                        print(result.stderr)
                    except Exception as e:
                        errors.append(f"{url}: {e}")
            else:
                ydl_opts = {
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en-US', 'en'],
                    'skip_download': True,
                    'outtmpl': output_template,
                    'quiet': True,
                    'subtitlesformat': 'srt'
                }
                if cookies_path:
                    ydl_opts['cookies'] = cookies_path
                try:
                    with YoutubeDL(ydl_opts) as ydl:
                        ydl.download(urls)
                except Exception as e:
                    errors.append(str(e))
        except Exception as e:
            errors.append(str(e))

        srt_files = glob.glob(os.path.join(downloads_dir, "*.srt"))
        vtt_files = glob.glob(os.path.join(downloads_dir, "*.vtt"))
        all_files = srt_files + vtt_files

        if not all_files:
            if any("HTTP Error 429" in e or "429" in e for e in errors):
                return None, "Error: HTTP 429 Too Many Requests from YouTube. Try again later."
            err_msg = "; ".join(errors) if errors else "No subtitle files found in Downloads."
            return None, f"SRT download error: {err_msg}"

        temp_dir = tempfile.mkdtemp(prefix="ssui_srt_")
        copied_paths = []
        copy_errors = []
        for fpath in all_files:
            try:
                dest = os.path.join(temp_dir, os.path.basename(fpath))
                shutil.copy2(fpath, dest)
                copied_paths.append(dest)
            except Exception as e:
                copy_errors.append(f"{fpath}: {e}")

        if not copied_paths:
            msg = "; ".join(copy_errors) if copy_errors else "Failed to copy subtitle files."
            return None, f"SRT copy error: {msg}"

        if len(copied_paths) == 1:
            return copied_paths[0], f"Downloaded subtitle copied to {copied_paths[0]}"

        zip_base = os.path.join(temp_dir, "srt_files")
        zip_path = shutil.make_archive(zip_base, "zip", temp_dir)
        return zip_path, f"Multiple subtitle files archived to {zip_path}"

    except Exception as e:
        print("SRT download error:", e)
        return None, "Saved in Downloads"