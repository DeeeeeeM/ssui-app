from yt_dlp import YoutubeDL
import csv

playlist_url = 'https://www.youtube.com/playlist?list=PLGRhcC_vtOra_TUIec1NgfHJIggPONtqU'

ydl_opts = {
    'extract_flat': True,
    'quiet': True,
    'dump_single_json': True
}

with YoutubeDL(ydl_opts) as ydl:
    result = ydl.extract_info(playlist_url, download=False)

    entries = result.get('entries', [])

    with open('playlist.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Title', 'Video ID', 'URL'])  # Header

        for video in entries:
            title = video.get('title', 'N/A')
            video_id = video['id']
            url = f'https://www.youtube.com/watch?v={video_id}'
            writer.writerow([title, video_id, url])

print("✅ Video IDs and URLs saved to 'playlist.csv'")
