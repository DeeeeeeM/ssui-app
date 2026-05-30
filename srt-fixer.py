import re
import sys
import json
from datetime import timedelta

# --- Helper functions for time conversion --- #

def srt_time_to_seconds(srt_time: str) -> float:
    """
    Convert SRT time (HH:MM:SS,ms) to seconds.
    """
    h, m, s_ms = srt_time.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def seconds_to_srt_time(seconds: float) -> str:
    """
    Convert seconds (float) back to SRT time format.
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    ms = int((seconds - int(seconds)) * 1000)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# --- Data structure --- #

class Caption:
    def __init__(self, index: int, start: str, end: str, text: str):
        self.index = index
        self.start_str = start
        self.end_str = end
        self.start = srt_time_to_seconds(start)
        self.end = srt_time_to_seconds(end)
        self.text = text.rstrip()  # Preserve line breaks

    @property
    def duration(self) -> float:
        return self.end - self.start

    def update_times(self):
        self.start_str = seconds_to_srt_time(self.start)
        self.end_str = seconds_to_srt_time(self.end)

    def __repr__(self):
        return f"{self.index}\n{self.start_str} --> {self.end_str}\n{self.text}\n"

# --- Parsing and formatting SRT --- #

def parse_srt(srt_content: str):
    """
    Parse SRT file content into a list of Caption objects.
    """
    captions = []
    # Split the content by blank lines (assuming each caption block is separated by one or more blank lines)
    blocks = srt_content.strip().split("\n\n")
    for block in blocks:
        lines = block.splitlines()
        if len(lines) >= 3:
            try:
                index = int(lines[0])
            except ValueError:
                continue
            # The second line should be the timing line.
            time_line = lines[1]
            if " --> " not in time_line:
                continue
            start, _, end = time_line.partition(" --> ")
            # Preserve the original multiline caption text.
            text = "\n".join(lines[2:])
            captions.append(Caption(index, start.strip(), end.strip(), text))
    return captions

def format_srt(captions):
    """
    Format a list of Caption objects back into SRT file format.
    """
    output = []
    for cap in captions:
        cap.update_times()
        output.append(f"{cap.index}\n{cap.start_str} --> {cap.end_str}\n{cap.text}\n")
    return "\n".join(output)

# --- Overlap detection and linear interpolation --- #

def adjust_overlapping_captions(captions):
    """
    Detect groups of overlapping captions and adjust their time ranges by evenly splitting the total time range.
    """
    if not captions:
        return

    groups = []
    current_group = [captions[0]]
    
    # Group consecutive captions if there is any overlap
    for cap in captions[1:]:
        # if the start of the next caption is before the end of the last one in the current group,
        # consider it overlapping.
        if cap.start < current_group[-1].end:
            current_group.append(cap)
        else:
            groups.append(current_group)
            current_group = [cap]
    groups.append(current_group)
    
    # Process each group
    for group in groups:
        if len(group) == 1:
            continue
        # Total available interval for the group is from the earliest start to the latest end.
        group_start = group[0].start
        group_end = group[-1].end
        total_duration = group_end - group_start
        n = len(group)
        # Evenly assign intervals
        for i, cap in enumerate(group):
            new_start = group_start + (i * total_duration) / n
            new_end = group_start + ((i + 1) * total_duration) / n
            print(f"Adjusted range for caption {cap.index} from {cap.start} {cap.end} -> {new_start} {new_end}")
            cap.start = new_start
            cap.end = new_end

# --- Heuristic to move dangling words --- #

def is_dangling_word(word: str) -> bool:
    """
    Heuristic: consider a word dangling if it is very short (<=5 characters) 
    and is alphabetical with an initial uppercase letter.
    """
    return word.isalpha() and len(word) <= 5 and word[0].isupper()

def move_dangling_words(captions):
    """
    For each caption (except the last), check the last word of its final line.
    If that word is "dangling", remove it from the current caption and prepend it
    to the next caption, preserving the multiline formatting.
    """
    for i in range(len(captions) - 1):
        cap = captions[i]
        # Split text into lines without discarding the structure
        lines = cap.text.splitlines()
        if not lines:
            continue
        # Process only the last line
        last_line = lines[-1].rstrip()
        if not last_line:
            continue
        words = last_line.split()
        if not words:
            continue
        last_word = words[-1]
        if is_dangling_word(last_word):
            # Remove the dangling word from the last line
            words = words[:-1]
            new_last_line = " ".join(words)
            lines[-1] = new_last_line
            cap.text = "\n".join(lines)
            
            # Prepend the dangling word to the first line of the next caption
            next_lines = captions[i+1].text.splitlines()
            if next_lines:
                next_lines[0] = last_word + " " + next_lines[0]
            else:
                next_lines = [last_word]
            captions[i+1].text = "\n".join(next_lines)
            print(f"Moved dangling word '{last_word}' from caption {cap.index} to {captions[i+1].index}.")

# --- Heuristic to check for too-short time ranges --- #

def check_time_ranges(captions, threshold=3.5):
    """
    For each caption, compute words per second. If it exceeds the threshold,
    print a warning to verify the caption manually.
    """
    warnings = []
    for cap in captions:
        word_count = len(cap.text.split())
        if cap.duration > 0:
            words_per_second = word_count / cap.duration
            if words_per_second > threshold:
                warning = (f"Caption {cap.index} may be too short for its text "
                           f"(duration: {cap.duration:.2f}s, words: {word_count}, {words_per_second:.2f} w/s).")
                warnings.append(warning)
                print(warning)
    return warnings

# --- Verification against JSON word timings --- #

def normalize_word(word):
    """
    Normalize a word by lowercasing and stripping non-alphanumeric characters.
    This helps align words from the SRT caption and the JSON word tokens.
    """
    return re.sub(r'[^a-z0-9]', '', word.lower())

def find_caption_word_window(caption, json_data):
    """
    Flatten all JSON word tokens and attempt to find a contiguous subsequence that
    matches the normalized caption text tokens.
    
    Returns a tuple (json_start, json_end) using the timing of the first and last matching word.
    If no exact match is found, returns None.
    """
    # Flatten JSON words from all segments in order.
    json_words = []
    for segment in json_data.get("segments", []):
        for word_data in segment.get("words", []):
            if "word" in word_data:
                json_words.append(word_data)
                
    # Create a list of normalized tokens from the JSON words.
    json_tokens = [normalize_word(wd["word"]) for wd in json_words]
    
    # Tokenize the caption text in a similar way.
    caption_tokens = [normalize_word(w) for w in caption.text.split()]
    
    n = len(json_tokens)
    m = len(caption_tokens)
    
    # Try to find an exact contiguous subsequence match.
    for i in range(n - m + 1):
        if json_tokens[i:i+m] == caption_tokens:
            # Found a matching window.
            window = json_words[i:i+m]
            
            # Determine json_start from the first word with a "start" timing.
            json_start = None
            for wd in window:
                if "start" in wd:
                    json_start = wd["start"]
                    break
            # Determine json_end: look from the last word backwards for an "end".
            json_end = None
            for wd in reversed(window):
                if "end" in wd:
                    json_end = wd["end"]
                    break
            # If the last word in the window lacks "end", extend the search beyond the window.
            if json_end is None:
                for wd in json_words[i+m:]:
                    if "end" in wd:
                        json_end = wd["end"]
                        break
            if json_start is not None and json_end is not None:
                return json_start, json_end
    # If no exact match is found, you could extend this to a fuzzy match.
    return None

def adjust_captions_with_json(captions, json_file_path, time_tolerance=0.6):
    """
    For each caption, adjust its start and end times to match the JSON word timings.
    Then, ensure that no caption overlaps with the next one.
    
    Parameters:
      captions: List of Caption objects.
      json_file_path: JSON data containing word timings.
      time_tolerance: A small tolerance (in seconds) to decide if an adjustment is needed.
      
    Returns:
      The list of adjusted Caption objects.
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    for i, cap in enumerate(captions):
        # Try to align the caption text against the JSON words.
        verified = find_caption_word_window(cap, json_data)
        if verified:
            json_start, json_end = verified
            # If there's a significant difference between the SRT times and JSON times, update.
            if abs(cap.start - json_start) > time_tolerance or abs(cap.end - json_end) > time_tolerance:
                print(f"Adjusting caption {cap.index}: SRT {cap.start:.2f}-{cap.end:.2f} -> JSON {json_start:.2f}-{json_end:.2f}")
                cap.start = json_start
                cap.end = json_end
        else:
            print(f"Warning: Caption {cap.index} could not be aligned with JSON words.")

        # Ensure the next caption does not overlap.
        if i < len(captions) - 1:
            next_cap = captions[i+1]
            if next_cap.start < cap.end:
                # Shift the next caption's start to the current caption's end.
                print(f"Fixing overlap: Caption {next_cap.index} start {next_cap.start:.2f} shifted to {cap.end:.2f}")
                next_cap.start = cap.end
                
    return captions


# --- Main processing function --- #

def process_srt_file(input_path, output_path, json_file_path):
    with open(input_path, "r", encoding="utf-8") as f:
        srt_content = f.read()
    
    captions = parse_srt(srt_content)
    
    # Adjust overlapping captions by linear interpolation.
    adjust_overlapping_captions(captions)
    
    # Apply heuristic to move dangling words.
    move_dangling_words(captions)
    
    # Check for too-short time ranges.
    check_time_ranges(captions)

    # Adjust captions using the JSON file.
    adjust_captions_with_json(captions, json_file_path)
    
    # Re-index captions if necessary
    for idx, cap in enumerate(captions, start=1):
        cap.index = idx
        
    output_srt = format_srt(captions)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_srt)
    
    print(f"Processed SRT saved to {output_path}")

# --- Main Script Usage --- #

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    json_file = sys.argv[3]
    process_srt_file(input_file, output_file, json_file)