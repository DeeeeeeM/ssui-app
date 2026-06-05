import subprocess
import csv
import sys
import json
import re
from urllib.parse import urlparse, parse_qs

SHORT_MIN = 0 * 60      # 0:00 – 7:00  → Short Form
SHORT_MAX = 7 * 60
MID_MIN   = 8 * 60      # 8:00 – 15:00 → Mid Form
MID_MAX   = 15 * 60

VIOLATOR_PATTERNS = [
    r"[?&]list=",          # playlist parameter
    r"/playlist\?",        # playlist page
    r"/@[^/]+/?$",         # channel home
    r"/channel/",          # channel URL
    r"/c/[^/]+/?$",        # custom channel
    r"/user/[^/]+/?$",     # legacy user channel
    r"/results\?",         # search results
    r"/feed/",             # feed pages
    r"youtu\.be/.{11}.+",  # youtu.be with extra params (playlist, etc.)
]

def is_violator(url: str) -> str | None:
    """Return a reason string if the URL is not a single video, else None."""
    for pattern in VIOLATOR_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            if "list=" in url and "v=" in url:
                return "Video in playlist — ambiguous"
            if "list=" in url:
                return "Playlist URL"
            if "/playlist?" in url:
                return "Playlist page"
            if re.search(r"/@|/channel/|/c/|/user/", url):
                return "Channel URL"
            if "/results?" in url:
                return "Search results URL"
            if "/feed/" in url:
                return "Feed URL"
            return "Non-single-video URL"
    return None


def fetch_video_info(url: str) -> dict | None:
    """Use yt-dlp to fetch duration, title, and upload year. Returns None on failure."""
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "--skip-download",
                "--quiet",
                "--no-warnings",
                "--dump-json",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(result.stderr.strip() or "yt-dlp returned no data")

        metadata = json.loads(result.stdout)
        duration = metadata.get("duration")
        title = metadata.get("title", "")
        upload_date = metadata.get("upload_date", "")
        year = upload_date[:4] if upload_date and len(upload_date) >= 4 else ""

        if duration is None:
            return None

        return {
            "duration_seconds": int(duration),
            "duration": seconds_to_hms(int(duration)),
            "title": title,
            "year": year,
        }
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError, RuntimeError, json.JSONDecodeError) as e:
        print(f"  [!] yt-dlp error for {url}: {e}", file=sys.stderr)
        return None


def seconds_to_hms(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def categorize(seconds: int) -> str | None:
    if SHORT_MIN <= seconds <= SHORT_MAX:
        return "Short Form"
    elif MID_MIN <= seconds <= MID_MAX:
        return "Mid Form"
    elif seconds >= MID_MAX:
        return "Could be Full Eps"
    return None


def process_urls(urls: list[str]) -> dict:
    results = {
        "Short Form": [],
        "Mid Form":   [],
        "Could be Full Eps": [],
        "Violators":  [],
    }

    for i, url in enumerate(urls, 1):
        url = url.strip()
        if not url:
            continue
        print(f"[{i}/{len(urls)}] Processing: {url}")

        # Check for violators 
        reason = is_violator(url)
        if reason:
            print(f"  → Violator: {reason}")
            results["Violators"].append({"url": url, "reason": reason})
            continue

        # Fetch metadata
        info = fetch_video_info(url)
        if info is None:
            print(f"  → Could not fetch metadata — marking as Violator")
            results["Violators"].append({"url": url, "reason": "Could not fetch metadata"})
            continue

        category = categorize(info["duration_seconds"])
        if category:
            print(f"  → {category} ({info['duration']})")
            results[category].append({
                "url": url,
                "title": info["title"],
                "year": info["year"],
                "duration": info["duration"],
            })
        else:
            print(f"  → Duration out of range ({info['duration']}) — marking as Violator")
            results["Violators"].append({"url": url, "reason": f"Duration out of range ({info['duration']})"})

    return results


def save_csv(results: dict, output_path: str):
    """
    Save results to CSV 
    """
    short = results["Short Form"]
    mid   = results["Mid Form"]
    full_eps = results["Could be Full Eps"]
    viol  = results["Violators"]

    max_rows = max(len(short), len(mid), len(full_eps), len(viol), 1)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "Short Form Title", "Short Form URL", "Short Form Year", "Short Form Duration",
            "Mid Form Title",   "Mid Form URL",   "Mid Form Year",   "Mid Form Duration",
            "Could be Full Eps Title", "Could be Full Eps URL", "Could be Full Eps Year", "Could be Full Eps Duration",
            "Violator URL",     "Violator Reason",
        ])

        for i in range(max_rows):
            s = short[i] if i < len(short) else {}
            m = mid[i]   if i < len(mid)   else {}
            f = full_eps[i] if i < len(full_eps) else {}
            v = viol[i]  if i < len(viol)  else {}

            writer.writerow([
                s.get("title", ""), s.get("url", ""), s.get("year", ""), s.get("duration", ""),
                m.get("title", ""), m.get("url", ""), m.get("year", ""), m.get("duration", ""),
                f.get("title", ""), f.get("url", ""), f.get("year", ""), f.get("duration", ""),
                v.get("url", ""),   v.get("reason", ""),
            ])

        # Summary 
        writer.writerow([])
        writer.writerow(["Summary"])
        writer.writerow(["Short Form Count", len(short)])
        writer.writerow(["Mid Form Count", len(mid)])
        writer.writerow(["Could be Full Eps Count", len(full_eps)])
        writer.writerow(["Violator Count", len(viol)])
        writer.writerow(["Processed Videos", len(short) + len(mid) + len(full_eps)])

    print(f"\n✅ Saved to: {output_path}")


def print_summary(results: dict):
    processed = len(results["Short Form"]) + len(results["Mid Form"]) + len(results["Could be Full Eps"])
    print("\n" + "═" * 50)
    print("  SUMMARY")
    print("═" * 50)
    print(f"  Short Form        (0-7 min)   : {len(results['Short Form'])} videos")
    print(f"  Mid Form          (8-15 min)  : {len(results['Mid Form'])} videos")
    print(f"  Could be Full Eps (≥15 min)   : {len(results['Could be Full Eps'])} videos")
    print(f"  Processed Videos               : {processed} videos")
    print(f"  Violators                     : {len(results['Violators'])} links")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    print("=" * 50)
    print("  YouTube Duration Filter (yt-dlp)")
    print("=" * 50)
    print()

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        with open(input_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(urls)} URLs from: {input_file}\n")
    else:
        print("Paste YouTube URLs (one per line).")
        print("Press Enter twice (blank line) when done:\n")
        urls = []
        while True:
            line = input()
            if not line:
                break
            urls.append(line.strip())

    if not urls:
        print("No URLs provided. Exiting.")
        sys.exit(0)

    output_csv = sys.argv[2] if len(sys.argv) > 2 else "yt_filtered.csv"

    results = process_urls(urls)
    print_summary(results)
    save_csv(results, output_csv)
