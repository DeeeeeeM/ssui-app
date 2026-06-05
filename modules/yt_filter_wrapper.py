"""
Wrapper for YT Filter functionality to integrate with Gradio.
Filters YouTube URLs by video duration and type.
"""

import tempfile
import os
from yt_dlp import YoutubeDL
from modules.utils import normalize_file_path
from modules.yt_filter import process_urls, save_csv


def _split_input_urls(urls_text):
    urls = []
    for line in urls_text.strip().split('\n'):
        for part in line.split(','):
            url = part.strip()
            if url and url.startswith(('http://', 'https://', 'youtube.com', 'youtu.be')):
                urls.append(url)
    return urls


def _extract_playlist_video_urls(playlist_url, cookies_path=None):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'dump_single_json': True,
    }
    cookies_path = normalize_file_path(cookies_path)
    if cookies_path:
        ydl_opts['cookiefile'] = cookies_path

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        entries = info.get('entries', []) or []
        urls = []
        for entry in entries:
            video_id = entry.get('id') or entry.get('url')
            if video_id:
                urls.append(f'https://www.youtube.com/watch?v={video_id}')
        return urls


def filter_youtube_urls(urls_text, input_mode='video', cookies_path=None):
    """
    Filter YouTube URLs by duration and type.
    
    Args:
        urls_text: Text containing URLs separated by newlines or commas
        input_mode: 'video' or 'playlist'
        cookies_path: Optional cookies file path for auth
    
    Returns:
        Tuple of (output_csv_path, summary_message)
    """
    if not urls_text or not urls_text.strip():
        return None, "❌ Error: Please provide at least one URL"
    
    try:
        urls = _split_input_urls(urls_text)
        if not urls:
            return None, "❌ Error: No valid YouTube URLs found. URLs should start with http(s):// or be youtube.com/youtu.be links"

        if input_mode == 'playlist':
            playlist_urls = urls
            expanded_urls = []
            for playlist_url in playlist_urls:
                videos = _extract_playlist_video_urls(playlist_url, cookies_path)
                if videos:
                    expanded_urls.extend(videos)
            if not expanded_urls:
                return None, "❌ Error: No videos found in provided playlist URLs"
            urls = expanded_urls
            source_label = f"Playlist input expanded to {len(urls)} videos"
        else:
            source_label = f"Video input processed ({len(urls)} URLs)"
        
        # Process URLs
        results = process_urls(urls)
        
        # Save to CSV
        fd, csv_path = tempfile.mkstemp(suffix=".csv", text=True)
        os.close(fd)
        save_csv(results, csv_path)
        
        # Generate summary
        short_count = len(results["Short Form"])
        mid_count = len(results["Mid Form"])
        full_count = len(results["Could be Full Eps"])
        violator_count = len(results["Violators"])
        total_processed = short_count + mid_count + full_count
        
        summary = (
            f"✅ Processing complete!\n\n"
            f"{source_label}\n"
            f"📊 Results:\n"
            f"  • Short Form (0-7 min): {short_count}\n"
            f"  • Mid Form (8-15 min): {mid_count}\n"
            f"  • Could be Full Eps (15+ min): {full_count}\n"
            f"  • Violators: {violator_count}\n\n"
            f"📈 Total videos processed: {total_processed}\n"
            f"Total URLs checked: {len(urls)}"
        )
        
        return csv_path, summary
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"
