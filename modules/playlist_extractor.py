from modules.utils import normalize_file_path

import os
import csv
import tempfile
from yt_dlp import YoutubeDL

def extract_playlist_to_csv(playlist_url, cookies_path=None):
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'dump_single_json': True
    }
    try:
        cookies_path = normalize_file_path(cookies_path)
        if cookies_path:
            ydl_opts['cookies'] = cookies_path
        with YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(playlist_url, download=False)
            entries = result.get('entries', [])
            fd, csv_path = tempfile.mkstemp(suffix=".csv", text=True)
            os.close(fd)
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Title', 'Video ID', 'URL'])
                for video in entries:
                    title = video.get('title', 'N/A')
                    video_id = video['id']
                    url = f'https://www.youtube.com/watch?v={video_id}'
                    writer.writerow([title, video_id, url])
        return csv_path
    except Exception as e:
        return None