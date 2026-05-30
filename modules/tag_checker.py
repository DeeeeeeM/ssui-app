from modules.utils import normalize_file_path

import os
from yt_dlp import YoutubeDL

def check_youtube_tag(video_url, tag_to_check, cookies_path=None):
    try:
        cookies_path = normalize_file_path(cookies_path)
        ydl_opts = {"quiet": True}
        if cookies_path:
            ydl_opts["cookies"] = cookies_path
        # Use a browser-like User-Agent by default to reduce SABR/format issues
        ydl_opts.setdefault("http_headers", {})
        ydl_opts["http_headers"].setdefault("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            tags = info.get('tags', [])
            tag_to_check_norm = tag_to_check.lower()
            tags_norm = [t.lower() for t in tags]
            # Exact match, case-insensitive, apostrophe style must match
            exists = any(tag_to_check_norm == t for t in tags_norm)
            if exists:
                return f"Tag/s '{tag_to_check}' EXISTS in video"
            else:
                return f"Tag/s '{tag_to_check}' DOES NOT EXIST in video.\n\nTags found: {tags if tags else 'None'}"
    except Exception as e:
        err = str(e)
        if 'Sign in to confirm your age' in err or ('Sign in' in err and 'age' in err):
            return f"Error checking {video_url}: This video is age-restricted and requires authentication (provide a cookies.txt file)."
        if 'HTTP Error 403' in err or '403' in err:
            return f"Error checking {video_url}: HTTP 403 Forbidden - try supplying a cookies file or updating yt-dlp with `yt-dlp -U`."
        return f"Error checking {video_url}: {err}"

def check_playlist_tags(playlist_url, tag_to_check, cookies_path=None):
    import tempfile, csv
    try:
        cookies_path = normalize_file_path(cookies_path)
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
            'dump_single_json': True
        }
        if cookies_path:
            ydl_opts['cookies'] = cookies_path
        # Use browser user agent
        ydl_opts.setdefault("http_headers", {})
        ydl_opts["http_headers"].setdefault("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        with YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(playlist_url, download=False)
            entries = result.get('entries', [])
            rows = []
            tag_to_check_norm = tag_to_check.lower()
            for video in entries:
                video_id = video.get('id')
                if not video_id:
                    title = video.get('title', 'N/A')
                    rows.append([title, '', 'No video ID in playlist entry'])
                    continue
                video_url = f'https://www.youtube.com/watch?v={video_id}'
                title = video.get('title', 'N/A')
                video_opts = {'quiet': True}
                if cookies_path:
                    video_opts['cookies'] = cookies_path
                # Add a user agent
                video_opts.setdefault("http_headers", {})
                video_opts["http_headers"].setdefault("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                try:
                    with YoutubeDL(video_opts) as ydl_video:
                        info = ydl_video.extract_info(video_url, download=False)
                        # Detect unlisted flag if available
                        is_unlisted = info.get('is_unlisted') if isinstance(info, dict) else False
                        # Detect private, membership or age-limit fields if present
                        is_private = info.get('is_private') if isinstance(info, dict) and 'is_private' in info else False
                        age_limit = info.get('age_limit') if isinstance(info, dict) and 'age_limit' in info else 0
                        # Tags processing
                        tags = info.get('tags', []) or []
                        tags_norm = [t.lower() for t in tags]
                        exists = any(tag_to_check_norm == t for t in tags_norm)
                        # Build note components
                        parts = []
                        if is_unlisted:
                            parts.append('Unlisted')
                        if is_private:
                            parts.append('Private')
                        elif age_limit and int(age_limit) >= 18:
                            parts.append('Age-restricted')
                        if exists:
                            parts.append(f"Tag/s '{tag_to_check}' exists in video")
                        else:
                            parts.append('Tag/s does not exist in video')
                        note = '; '.join(parts)
                        rows.append([title, video_url, note])
                except Exception as e:
                    err = str(e)
                    err_lower = err.lower()
                    if 'sign in to confirm your age' in err_lower or ('age' in err_lower and 'sign in' in err_lower):
                        note = 'Age-restricted - cookies required or signed-in account needed'
                    elif 'private' in err_lower and 'video' in err_lower:
                        note = 'Private video - access denied'
                    elif 'video unavailable' in err_lower or 'not available' in err_lower or 'removed' in err_lower:
                        note = 'Video unavailable or removed'
                    elif '403' in err_lower or 'forbidden' in err_lower:
                        note = 'HTTP Error 403 Forbidden - cookies may be required or access denied'
                    else:
                        note = f"Could not check video: {err}"
                    rows.append([title, video_url, note])
            # Write to temp CSV
            fd, csv_path = tempfile.mkstemp(suffix=".csv", text=True)
            os.close(fd)
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Title", "URL", "Notes"])
                writer.writerows(rows)
            return csv_path
    except Exception as e:
        # Write error to CSV
        fd, csv_path = tempfile.mkstemp(suffix=".csv", text=True)
        os.close(fd)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Title", "URL", "Notes"])
            writer.writerow(["Error", "", str(e)])
        return csv_path