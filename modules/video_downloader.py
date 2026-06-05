"""
YouTube video and audio downloader with quality choices.
Supports downloading single videos and playlists in mp3 and mp4 formats.
"""

import os
import tempfile
from yt_dlp import YoutubeDL


def download_single_video(url, format_type, quality, cookies_path=None):
    """
    Download a single YouTube video or audio.
    
    Args:
        url: YouTube video URL
        format_type: "mp3" or "mp4"
        quality: "high", "medium", "low"
        cookies_path: Path to cookies file (optional)
    
    Returns:
        Tuple of (output_file_path, status_message)
    """
    if not url or not url.strip():
        return None, "❌ Error: Please provide a video URL"
    
    try:
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        
        # Quality settings
        quality_settings = {
            "high": {
                "mp3": {"format": "bestaudio/best", "audio_quality": 0},
                "mp4": {"format": "best[ext=mp4]"}
            },
            "medium": {
                "mp3": {"format": "bestaudio/best", "audio_quality": 5},
                "mp4": {"format": "worst[ext=mp4]"}  # or use best[height<=720]
            },
            "low": {
                "mp3": {"format": "bestaudio/best", "audio_quality": 9},
                "mp4": {"format": "best[height<=480]"}
            }
        }
        
        settings = quality_settings.get(quality, quality_settings["medium"])
        
        if format_type == "mp3":
            ydl_opts = {
                "format": settings["mp3"]["format"],
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": str(settings["mp3"]["audio_quality"]),
                    }
                ],
                "outtmpl": os.path.join(downloads_dir, "%(title)s.%(ext)s"),
                "quiet": False,
                "no_warnings": False,
            }
            if cookies_path:
                ydl_opts["cookiefile"] = cookies_path
            
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                base_name = os.path.splitext(filename)[0]
                output_file = f"{base_name}.mp3"
                
                if os.path.exists(output_file):
                    return output_file, f"✅ Downloaded: {info.get('title', 'Unknown')} (mp3)"
                else:
                    return None, "❌ Error: MP3 file was not created"
        
        elif format_type == "mp4":
            ydl_opts = {
                "format": settings["mp4"]["format"],
                "outtmpl": os.path.join(downloads_dir, "%(title)s.%(ext)s"),
                "quiet": False,
                "no_warnings": False,
            }
            if cookies_path:
                ydl_opts["cookiefile"] = cookies_path
            
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                
                if os.path.exists(filename):
                    return filename, f"✅ Downloaded: {info.get('title', 'Unknown')} (mp4)"
                else:
                    return None, "❌ Error: MP4 file was not created"
        
        return None, "❌ Error: Invalid format type"
        
    except Exception as e:
        return None, f"❌ Download error: {str(e)}"


def download_playlist(url, format_type, quality, cookies_path=None):
    """
    Download all videos from a YouTube playlist.
    
    Args:
        url: YouTube playlist URL
        format_type: "mp3" or "mp4"
        quality: "high", "medium", "low"
        cookies_path: Path to cookies file (optional)
    
    Returns:
        Tuple of (output_folder_path, status_message)
    """
    if not url or not url.strip():
        return None, "❌ Error: Please provide a playlist URL"
    
    try:
        base_downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        playlist_dir = os.path.join(base_downloads_dir, "playlist_download")
        os.makedirs(playlist_dir, exist_ok=True)
        
        # Quality settings
        quality_settings = {
            "high": {
                "mp3": {"format": "bestaudio/best", "audio_quality": 0},
                "mp4": {"format": "best[ext=mp4]"}
            },
            "medium": {
                "mp3": {"format": "bestaudio/best", "audio_quality": 5},
                "mp4": {"format": "worst[ext=mp4]"}
            },
            "low": {
                "mp3": {"format": "bestaudio/best", "audio_quality": 9},
                "mp4": {"format": "best[height<=480]"}
            }
        }
        
        settings = quality_settings.get(quality, quality_settings["medium"])
        
        if format_type == "mp3":
            ydl_opts = {
                "format": settings["mp3"]["format"],
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": str(settings["mp3"]["audio_quality"]),
                    }
                ],
                "outtmpl": os.path.join(playlist_dir, "%(playlist)s", "%(title)s.%(ext)s"),
                "quiet": False,
                "no_warnings": False,
            }
            if cookies_path:
                ydl_opts["cookiefile"] = cookies_path
            
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                playlist_name = info.get("title", "Playlist")
                count = len(info.get("entries", []))
                return playlist_dir, f"✅ Downloaded {count} mp3 files from: {playlist_name}"
        
        elif format_type == "mp4":
            ydl_opts = {
                "format": settings["mp4"]["format"],
                "outtmpl": os.path.join(playlist_dir, "%(playlist)s", "%(title)s.%(ext)s"),
                "quiet": False,
                "no_warnings": False,
            }
            if cookies_path:
                ydl_opts["cookiefile"] = cookies_path
            
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                playlist_name = info.get("title", "Playlist")
                count = len(info.get("entries", []))
                return playlist_dir, f"✅ Downloaded {count} mp4 files from: {playlist_name}"
        
        return None, "❌ Error: Invalid format type"
        
    except Exception as e:
        return None, f"❌ Download error: {str(e)}"
