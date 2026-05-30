import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import re
import tempfile
import mimetypes
import gradio as gr
import torch
import stable_whisper
from stable_whisper.text_output import result_to_any, sec2srt
import time
from yt_dlp import YoutubeDL
import csv
import subprocess
import glob
import shutil

# ─────────────────────────────────────────────
#  NETFLIX STYLE CONSTANTS
# ─────────────────────────────────────────────
NETFLIX_MAX_CPS      = 17
NETFLIX_MAX_CPL      = 42
NETFLIX_MAX_LINES    = 2
NETFLIX_MIN_DURATION = 0.2
NETFLIX_MAX_DURATION = 7.0
NETFLIX_MIN_GAP      = 0.083   # ~2 frames @ 24 fps

# ─────────────────────────────────────────────
#  DEFAULT TAGLISH INITIAL PROMPT
# ─────────────────────────────────────────────
DEFAULT_TAGLISH_PROMPT = (
    "Tagalog at Ingles na pananalita. Filipino YouTube content. "
    "Ang code-switching sa Filipino at English ay karaniwan. "
    "Halimbawa ng mga salita: yung, kasi, naman, po, opo, diba, talaga, "
    "ganun, parang, syempre, pero, kaya, ano, eh, ba, nga."
)

# ─────────────────────────────────────────────
#  PRE-PROCESSING
# ─────────────────────────────────────────────

def preprocess_audio(input_path: str) -> str:
    """
    Normalize loudness, apply spectral noise filter, and resample to
    16 kHz mono WAV — Whisper's native format.
    Returns path to processed temp file.
    """
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-af", "loudnorm,afftdn=nf=-25",
        out_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        # ffmpeg not available or failed — fall back to original
        return input_path
    return out_path


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
#  MISSED SEGMENT DETECTION & RECOVERY
# ─────────────────────────────────────────────

def find_missed_segments(result, audio_duration: float, min_gap: float = 2.0):
    """Return list of (start, end) tuples for gaps longer than min_gap seconds."""
    missed = []
    segments = list(result)
    if not segments:
        return missed

    if segments[0].start > min_gap:
        missed.append((0.0, segments[0].start))

    for i in range(len(segments) - 1):
        gap_start = segments[i].end
        gap_end   = segments[i + 1].start
        if gap_end - gap_start > min_gap:
            missed.append((gap_start, gap_end))

    if segments[-1].end < audio_duration - min_gap:
        missed.append((segments[-1].end, audio_duration))

    return missed


def retry_missed_segments(model, audio_path: str, missed_segments, language: str,
                           model_type: str):
    """
    Re-run transcription on detected gaps with looser / more sensitive settings.
    Returns list of recovered segment-like objects with adjusted timestamps.
    """
    if not missed_segments:
        return []

    try:
        import librosa
        import soundfile as sf
    except ImportError:
        return []

    recovered = []
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception:
        return []

    for start, end in missed_segments:
        chunk = audio[int(start * sr): int(end * sr)]
        if len(chunk) < sr * 0.3:          # skip chunks < 300 ms
            continue

        fd, chunk_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(chunk_path, chunk, sr)

            if model_type == "faster whisper":
                chunk_result = model.transcribe(
                    chunk_path,
                    language=language,
                    vad=False,
                    no_speech_threshold=0.3,
                    condition_on_previous_text=False,
                    beam_size=10,
                    regroup=False,
                )
            else:
                chunk_result = model.transcribe(
                    chunk_path,
                    language=language,
                    vad=False,
                    no_speech_threshold=0.3,
                    condition_on_previous_text=False,
                    regroup=False,
                )

            for seg in chunk_result:
                seg.start += start
                seg.end   += start
                recovered.append(seg)
        except Exception:
            pass
        finally:
            try:
                os.unlink(chunk_path)
            except Exception:
                pass

    return recovered


# ─────────────────────────────────────────────
#  HALLUCINATION FILTER
# ─────────────────────────────────────────────

HALLUCINATION_PHRASES = [
    # English YouTube hallucinations
    "Thanks for watching", "Thank you for watching",
    "Please subscribe", "Like and subscribe",
    "Don't forget to subscribe", "Hit the subscribe button",
    "Subtitles by", "Transcribed by",
    "www.", ".com",
    # Filipino hallucinations
    "Salamat sa panood", "Mag-subscribe na",
    "I-like ang video", "Nag-subscribe ka na ba",
    "Huwag kalimutang mag-subscribe",
]

def clean_text(text: str) -> str:
    """
    Apply hallucination removal and repeated-phrase cleanup to a plain string.
    This operates only on text — never touches stable-ts Segment objects directly.
    """
    # Remove 3+ consecutive repeated word sequences (Whisper loop artifact)
    text = re.sub(
        r'\b(\w+(?:\s+\w+){0,4})\b(?:\s+\1){2,}',
        r'\1', text, flags=re.IGNORECASE
    )
    for phrase in HALLUCINATION_PHRASES:
        text = re.sub(re.escape(phrase), '', text, flags=re.IGNORECASE)
    return text.strip()


# ─────────────────────────────────────────────
#  TAGALOG PUNCTUATION FIXER
# ─────────────────────────────────────────────

TAGALOG_DISCOURSE_MARKERS = (
    r'\b(kasi|naman|diba|talaga|parang|syempre|pero|kaya|'
    r'yung|yun|nga|eh|ba|ano|po|opo|ganun|ganon)\b'
)

def fix_tagalog_punctuation(text: str) -> str:
    """Add light punctuation around common Tagalog discourse markers."""
    text = re.sub(
        TAGALOG_DISCOURSE_MARKERS + r'(?![,\.!?])',
        r'\1,', text, flags=re.IGNORECASE
    )
    text = re.sub(r'([\.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
    return text


def apply_text_filters_to_result(result, enable_hallucination_filter: bool,
                                  enable_punctuation_fix: bool, source_lang: str):
    """
    stable-ts Segment.text is a READ-ONLY property derived from its child WordTiming
    objects. To modify transcript text we must patch each word's `.word` attribute,
    then let stable-ts recompute `.text` naturally.

    Strategy:
      1. Collect the full segment text from words.
      2. Run our string filters on it.
      3. Redistribute the cleaned text back onto the word tokens proportionally,
         preserving word count. If the word count changes (phrase was removed) we
         blank out the surplus word slots so they become empty/whitespace and are
         effectively invisible in the output.
    """
    for seg in result.segments:
        if not seg.words:
            continue

        # Build current text from words (preserves original spacing/casing)
        original_words = [w.word for w in seg.words]
        current_text   = "".join(original_words).strip()

        modified_text = current_text

        if enable_hallucination_filter:
            modified_text = clean_text(modified_text)

        if enable_punctuation_fix and source_lang == "tl":
            modified_text = fix_tagalog_punctuation(modified_text)

        if modified_text == current_text:
            continue  # nothing changed — skip

        # Split cleaned text back into tokens
        new_tokens = modified_text.split()
        old_tokens = [w.strip() for w in original_words if w.strip()]

        for i, word_obj in enumerate(seg.words):
            if i < len(new_tokens):
                # Preserve leading/trailing whitespace from original slot
                leading  = " " if word_obj.word.startswith(" ") else ""
                trailing = " " if word_obj.word.endswith(" ")   else ""
                word_obj.word = leading + new_tokens[i] + trailing
            else:
                # Surplus slot — blank it out (becomes invisible in SRT output)
                word_obj.word = ""


# ─────────────────────────────────────────────
#  NETFLIX STYLE GUIDE ENFORCEMENT
# ─────────────────────────────────────────────

def get_cps(text: str, duration: float) -> float:
    chars = len(text.replace('\n', '').replace(' ', ''))
    return chars / duration if duration > 0 else 0.0


def enforce_netflix_style(result,
                          max_lines_per_segment=2,
                          line_penalty=22.01,
                          longest_line_char_penalty=1.0) -> tuple:
    """
    Checks segments against the Netflix timed-text style guide.
    IMPORTANT: CPL and line-count checks are done on the RENDERED text
    (after optimize_text applies smart line breaks), not on the raw
    segment string — which has no newlines yet when this runs.

    Gap violations ignore exact 0.000s boundaries (shared timestamps between
    back-to-back subtitles are normal and not a real gap problem).

    Returns (result, violations_report_string).
    """
    violations = []
    segs = list(result)

    for i, seg in enumerate(segs):
        duration = seg.end - seg.start
        raw_text = seg.text.strip()

        # Render the text exactly as it will appear in the SRT
        # (with smart line breaks applied) so CPL/line checks are accurate
        rendered = optimize_text(
            raw_text,
            max_lines_per_segment,
            line_penalty,
            longest_line_char_penalty
        )
        lines = rendered.split('\n')
        cps   = get_cps(raw_text, duration)
        issues = []

        # --- CPS ---
        if cps > NETFLIX_MAX_CPS:
            issues.append(f"CPS {cps:.1f} > {NETFLIX_MAX_CPS}")
            needed = len(raw_text.replace(' ', '')) / NETFLIX_MAX_CPS
            seg.end = seg.start + min(needed, NETFLIX_MAX_DURATION)

        # --- Duration ---
        if duration < NETFLIX_MIN_DURATION:
            issues.append(f"Duration {duration:.2f}s < {NETFLIX_MIN_DURATION}s")
            seg.end = seg.start + NETFLIX_MIN_DURATION

        if duration > NETFLIX_MAX_DURATION:
            issues.append(f"Duration {duration:.2f}s > {NETFLIX_MAX_DURATION}s")

        # --- Line count (on rendered text) ---
        if len(lines) > NETFLIX_MAX_LINES:
            issues.append(f"{len(lines)} lines > max {NETFLIX_MAX_LINES}")

        # --- CPL (on rendered lines, not raw string) ---
        for li, line in enumerate(lines):
            if len(line) > NETFLIX_MAX_CPL:
                issues.append(f"Line {li+1}: {len(line)} chars > {NETFLIX_MAX_CPL}")

        # --- Gap to next ---
        # Skip exact 0.000s gaps — those are shared timestamp boundaries,
        # not real violations. Only flag genuine overlaps or sub-2-frame gaps
        # where the segments are truly distinct (gap between 0 and MIN_GAP).
        if i < len(segs) - 1:
            gap = segs[i + 1].start - seg.end
            if 0 < gap < NETFLIX_MIN_GAP:
                issues.append(f"Gap to next {gap:.3f}s < {NETFLIX_MIN_GAP}s")
            elif gap < 0:
                issues.append(f"Overlap with next {abs(gap):.3f}s")

        if issues:
            violations.append({
                'index': i,
                'start': seg.start,
                'text':  rendered[:50],
                'issues': issues
            })

    if not violations:
        report = "✅ All segments pass Netflix style guide."
    else:
        lines_out = [f"⚠️ {len(violations)} segment(s) with style issues:\n"]
        for v in violations[:15]:
            ts = f"{v['start']:.1f}s"
            lines_out.append(f"  [{ts}] \"{v['text']}...\"")
            for iss in v['issues']:
                lines_out.append(f"    • {iss}")
        if len(violations) > 15:
            lines_out.append(f"  ... and {len(violations) - 15} more.")
        report = '\n'.join(lines_out)

    return result, report


# ─────────────────────────────────────────────
#  OPTIMIZE TEXT  (linguistically-aware line breaker)
# ─────────────────────────────────────────────

def _break_penalty(words: list, split_after: int) -> float:
    """
    Return an extra penalty for breaking AFTER words[split_after - 1].
    Lower  = better break point.
    Higher = worse break point (mid-phrase, proper noun join, etc.)

    Rules (applied in priority order):
      - Breaking after sentence-ending punctuation (.!?) → big reward (negative penalty)
      - Breaking after a comma → small reward
      - Breaking between two Title-Case words → heavy penalty (proper noun / place name)
      - Breaking after a conjunction or preposition → moderate penalty
      - Otherwise → no extra penalty
    """
    if split_after <= 0 or split_after >= len(words):
        return 0.0

    prev_word = words[split_after - 1]   # word just before the break
    next_word = words[split_after]        # word just after the break

    # Sentence-ending punctuation — ideal break point
    if prev_word and prev_word[-1] in '.!?':
        return -18.0

    # Comma — good break point
    if prev_word and prev_word[-1] == ',':
        return -6.0

    # Both words are Title-Case → likely a proper noun pair (e.g. "Tondo, Maynila")
    # Penalize heavily so we never split "Tondo," / "Maynila." across lines
    if (prev_word and prev_word[0].isupper() and
            next_word and next_word[0].isupper()):
        return 30.0

    # Tagalog / English conjunctions and prepositions — prefer not to break after these
    BAD_BREAK_AFTER = {
        'ang', 'ng', 'sa', 'na', 'at', 'ay', 'ni', 'para', 'kung',
        'dahil', 'pero', 'kaya', 'habang', 'kahit', 'mula', 'tungkol',
        'the', 'a', 'an', 'of', 'in', 'on', 'at', 'by', 'for', 'to',
        'and', 'or', 'but', 'with', 'from',
    }
    if prev_word.lower().rstrip('.,!?') in BAD_BREAK_AFTER:
        return 12.0

    # Breaking before a Title-Case word that follows a lower-case word
    # (mid-sentence before a proper noun) — slight penalty
    if next_word and next_word[0].isupper() and prev_word and prev_word[0].islower():
        return 8.0

    return 0.0


def optimize_text(text, max_lines_per_segment=2, line_penalty=22.01,
                  longest_line_char_penalty=1.0):
    """
    Smart 2-line breaker for subtitle text.

    Goal: split text into at most max_lines_per_segment lines where
    each line is as close to NETFLIX_MAX_CPL (42) chars as possible.

    Strategy (in priority order):
      1. Find the split point closest to the char midpoint of the text
         that also falls on a good linguistic boundary (.!? > , > space)
      2. Among equally-good splits, prefer the one with the most
         balanced line lengths (minimise |len1 - len2|)
      3. Always produce a split — never return unsplit long text
    """
    CPL      = NETFLIX_MAX_CPL   # 42
    text     = text.strip()
    words    = text.split()
    n        = len(words)
    total    = len(text)

    # Single word or single-line request — return as-is
    if n <= 1 or max_lines_per_segment == 1:
        return text

    # Build prefix char lengths so we can find the char position of each word boundary
    # prefix[i] = char length of words[0..i-1] joined with spaces
    prefix = [0] * (n + 1)
    for i, w in enumerate(words):
        prefix[i + 1] = prefix[i] + len(w) + 1   # +1 for space

    target = total / 2   # ideal char position for the split

    best_split_idx = None
    best_score     = 10**30

    for split_at in range(1, n):   # split BEFORE words[split_at]
        line1_len = prefix[split_at] - 1          # chars in line 1
        line2_len = total - prefix[split_at]      # chars in line 2 (no leading space)

        # Never produce a line shorter than 10 chars (avoid "Maki?" getting split)
        if line1_len < 10 or line2_len < 10:
            continue

        # Primary score: closeness of line1 to the target midpoint
        balance   = abs(line1_len - target)

        # Linguistic bonus: reward splitting at punctuation / penalise bad spots
        prev_word = words[split_at - 1]
        next_word = words[split_at]
        ling = 0.0
        if prev_word and prev_word[-1] in '.!?':
            ling -= 20.0    # sentence end = best split
        elif prev_word and prev_word[-1] == ',':
            ling -= 8.0     # comma = good split
        # Penalise splitting between Title-Case words (proper nouns)
        if prev_word and prev_word[0].isupper() and next_word and next_word[0].isupper():
            ling += 25.0
        # Penalise splitting after short function words
        BAD = {'ang','ng','sa','na','ni','ay','at','o',
               'the','a','an','of','in','on','by','to','and','or'}
        if prev_word.lower().rstrip('.,!?') in BAD:
            ling += 15.0

        score = balance + ling
        if score < best_score:
            best_score     = score
            best_split_idx = split_at

    # Guaranteed fallback — balanced word midpoint
    if best_split_idx is None:
        best_split_idx = n // 2

    line1 = ' '.join(words[:best_split_idx])
    line2 = ' '.join(words[best_split_idx:])
    return f'{line1}\n{line2}'


def segment2optimizedsrtblock(segment: dict, idx: int, max_lines_per_segment,
                               line_penalty, longest_line_char_penalty, strip=True) -> str:
    text = segment["text"].strip() if strip else segment["text"]
    # Only apply smart line-breaking when text exceeds a single Netflix line (42 chars).
    # Short segments stay as one line — no unnecessary breaks.
    if len(text) > NETFLIX_MAX_CPL:
        rendered = optimize_text(text, max_lines_per_segment, line_penalty, longest_line_char_penalty)
    else:
        rendered = text
    return f'{idx}\n{sec2srt(segment["start"])} --> {sec2srt(segment["end"])}\n{rendered}'


def segments2blocks(segments, max_lines_per_segment, line_penalty, longest_line_char_penalty):
    return '\n\n'.join(
        segment2optimizedsrtblock(s, i, max_lines_per_segment, line_penalty,
                                  longest_line_char_penalty, strip=True)
        for i, s in enumerate(segments)
    )


# ─────────────────────────────────────────────
#  SHORT SEGMENT ABSORBER
# ─────────────────────────────────────────────

# Genuine short utterances that should NEVER be merged — they are complete
# on their own even though they are short.
STANDALONE_SHORT = re.compile(
    r'^(tama ba|oo|hindi|bakit|ano|hoy|hala|grabe|talaga|sige|sana|'
    r'opo|hindi po|oo po|salamat|pasensya|sorry|okay|ok|'
    r'[A-Z][a-z]+[!?.]?)[\s!?\.]*$',
    re.IGNORECASE
)

def _starts_new_sentence(text: str) -> bool:
    """
    Return True if this text begins a new independent sentence.
    Signals: starts with a capital letter that is NOT a continuation
    of the previous segment (i.e. not a proper noun mid-sentence).
    Used to BLOCK merges that would join two separate sentences.
    """
    text = text.strip()
    if not text:
        return False
    # Any text starting with a capital letter is treated as a new sentence start.
    # Exception: single capital letters like "I" in English are always new.
    return text[0].isupper()


def _prev_ends_sentence(text: str) -> bool:
    """Return True if text ends with terminal sentence punctuation."""
    text = text.strip()
    return bool(text) and text[-1] in SENTENCE_END_CHARS



def check_netflix_style(segments: list,
                        max_lines_per_segment=2,
                        line_penalty=22.01,
                        longest_line_char_penalty=1.0) -> str:
    """
    Netflix style guide checker for smart_segs dict list.
    Checks rendered text (after optimize_text) for CPL, lines, duration, CPS.
    Returns a report string.
    """
    violations = []

    for i, seg in enumerate(segments):
        duration = seg['end'] - seg['start']
        raw_text = seg['text'].strip()

        rendered = optimize_text(raw_text, max_lines_per_segment, line_penalty, longest_line_char_penalty)
        lines    = rendered.split('\n')
        cps      = get_cps(raw_text, duration)
        issues   = []

        if cps > NETFLIX_MAX_CPS:
            issues.append(f"CPS {cps:.1f} > {NETFLIX_MAX_CPS}")
        if duration < NETFLIX_MIN_DURATION:
            issues.append(f"Duration {duration:.2f}s < {NETFLIX_MIN_DURATION}s")
        if duration > NETFLIX_MAX_DURATION:
            issues.append(f"Duration {duration:.2f}s > {NETFLIX_MAX_DURATION}s")
        if len(lines) > NETFLIX_MAX_LINES:
            issues.append(f"{len(lines)} lines > max {NETFLIX_MAX_LINES}")
        for li, line in enumerate(lines):
            if len(line) > NETFLIX_MAX_CPL:
                issues.append(f"Line {li+1}: {len(line)} chars > {NETFLIX_MAX_CPL}")
        if i < len(segments) - 1:
            gap = segments[i + 1]['start'] - seg['end']
            if 0 < gap < NETFLIX_MIN_GAP:
                issues.append(f"Gap {gap:.3f}s < {NETFLIX_MIN_GAP}s")
            elif gap < 0:
                issues.append(f"Overlap {abs(gap):.3f}s")

        if issues:
            violations.append({'index': i, 'start': seg['start'],
                                'text': rendered[:50], 'issues': issues})

    if not violations:
        return "✅ All segments pass Netflix style guide."

    lines_out = [f"⚠️ {len(violations)} segment(s) with style issues:\n"]
    for v in violations[:15]:
        lines_out.append(f"  [{v['start']:.1f}s] \"{v['text']}...\"")
        for iss in v['issues']:
            lines_out.append(f"    • {iss}")
    if len(violations) > 15:
        lines_out.append(f"  ... and {len(violations) - 15} more.")
    return '\n'.join(lines_out)


def merge_short_segments(result, min_chars: int = 25, max_merge_gap: float = 1.5):
    """
    After split_by_length, absorb segments that are too short to stand alone.
    Strict rules to prevent cross-sentence merging:
      - NEVER merge if the previous segment ends with .!? (sentence boundary)
      - NEVER merge backward if current segment starts a new sentence (capital letter)
      - NEVER merge forward if the next segment starts a new sentence
      - NEVER merge standalone short utterances (Tama ba?, Alina!, Oo, etc.)
    """
    segs = result.segments
    i = 0
    while i < len(segs):
        seg      = segs[i]
        seg_text = _seg_text(seg).strip()
        char_len = len(seg_text)

        is_short      = char_len < min_chars
        is_standalone = bool(STANDALONE_SHORT.match(seg_text))

        if is_short and not is_standalone and len(segs) > 1:
            gap_prev = (seg.start - segs[i-1].end) if i > 0           else 9999
            gap_next = (segs[i+1].start - seg.end) if i < len(segs)-1 else 9999

            merged = False

            # --- Try merging BACKWARD into previous segment ---
            if i > 0 and gap_prev <= max_merge_gap:
                prev_text = _seg_text(segs[i-1])
                prev_ends_sentence  = _prev_ends_sentence(prev_text)
                cur_starts_sentence = _starts_new_sentence(seg_text)
                combined_ok = len(prev_text) + 1 + char_len <= 140

                # Block: prev ends a sentence AND current starts a new one
                cross_boundary = prev_ends_sentence and cur_starts_sentence

                if combined_ok and not cross_boundary:
                    _do_merge(segs[i-1], seg)
                    segs.pop(i)
                    i = max(0, i - 1)
                    merged = True

            # --- Try merging FORWARD into next segment ---
            if not merged and i < len(segs)-1 and gap_next <= max_merge_gap:
                nxt_text = _seg_text(segs[i+1])
                cur_ends_sentence   = _prev_ends_sentence(seg_text)
                nxt_starts_sentence = _starts_new_sentence(nxt_text)
                combined_ok = char_len + 1 + len(nxt_text) <= 140

                # Block: current ends a sentence AND next starts a new one
                cross_boundary = cur_ends_sentence and nxt_starts_sentence

                if combined_ok and not cross_boundary:
                    _do_merge(seg, segs[i+1])
                    segs.pop(i+1)
                    merged = True

            if not merged:
                i += 1
        else:
            i += 1

    return result


# ─────────────────────────────────────────────
#  SENTENCE BOUNDARY MERGER
# ─────────────────────────────────────────────

SENTENCE_END_CHARS = set('.!?')

# ---------------------------------------------------------------------------
# TWO-TIER dangling word system:
#
# HARD danglers — grammatically impossible as sentence endings.
# Trigger merge regardless of segment length.
# These are function words / prepositions that MUST be followed by content.
#   e.g. "ng", "sa", "ni", "para", "kung" — a sentence cannot end here.
#
# SOFT danglers — common in conversational Taglish but CAN end a subtitle card.
# Only trigger merge when segment is SHORT (< SOFT_DANGLING_MAX_CHARS).
#   e.g. "yung", "natin", "nga", "pero, yun nga" — fine as a card ending
#   when the segment is already 35+ chars (speaker is mid-thought but readable).
# ---------------------------------------------------------------------------

SOFT_DANGLING_MAX_CHARS = 30   # soft danglers only trigger merge below this length

# Hard danglers: prepositions/linkers that literally cannot end a clause
TAGALOG_HARD_DANGLERS = {
    # Core prepositions / case markers — never end a sentence
    'ng', 'sa', 'ni', 'nang',
    # Subordinating conjunctions — always open a dependent clause
    'para', 'kung', 'habang', 'kahit', 'upang', 'kapag', 'pag',
    'dahil', 'kasi', 'bilang', 'tungkol', 'laban', 'ayon', 'base',
    # Multi-word hard danglers
    'laban dito', 'laban sa', 'ayon sa', 'base sa',
    'wala pa', 'wala pang', 'hindi pa',
}

# Soft danglers: particles that trail off but can legitimately end a card
# when the segment is already long enough to be readable on its own
TAGALOG_SOFT_DANGLERS = {
    # Focus / topic markers
    'ang', 'na', 'ay',
    # Common discourse particles — only merge when segment is very short
    'pa', 'rin', 'din', 'raw', 'daw', 'po', 'nga', 'ba', 'kaya',
    'lang', 'lamang', 'muna', 'pala', 'naman', 'talaga', 'sana',
    # NOTE: 'yung', 'yun', 'natin', 'namin', 'nila', 'niya', 'ating' intentionally
    # excluded — these are extremely common Taglish mid-card endings and should
    # NOT trigger a merge even when segment is short. They produce correct-looking
    # subtitle cards like "as we prepare yung / ating iba pang gulay."
    # Conjunctions that can end a card mid-flow in Taglish
    'at', 'o', 'pero',
    # Multi-word soft danglers
    'pa rin', 'pa din', 'rin naman', 'din naman', 'pa naman',
    'daw po', 'raw po', 'nga po', 'lang po', 'muna po',
    'hindi naman', 'tuloy daw', 'tuloy pa', 'tuloy na',
}

# Combined for backward compat — used by _ends_with_dangling with length context
TAGALOG_DANGLING_ENDINGS = TAGALOG_HARD_DANGLERS | TAGALOG_SOFT_DANGLERS

def _seg_text(seg) -> str:
    """Get clean text from a stable-ts segment regardless of internal structure."""
    try:
        return seg.text.strip()
    except Exception:
        if seg.words:
            return "".join(w.word for w in seg.words).strip()
        return ""


def _get_trailing_phrase(text: str, n: int) -> str:
    """Return the last n words of text joined, stripped of punctuation."""
    words = text.lower().split()
    if len(words) < n:
        return ""
    return ' '.join(w.strip('.,!?;:') for w in words[-n:])


def _ends_with_hard_dangler(text: str) -> bool:
    """Return True if text ends with a HARD dangling word — always requires merge."""
    for n in (1, 2, 3):
        phrase = _get_trailing_phrase(text, n)
        if phrase and phrase in TAGALOG_HARD_DANGLERS:
            return True
    return False


def _ends_with_soft_dangler(text: str) -> bool:
    """Return True if text ends with a SOFT dangling word — only merge if segment is short."""
    for n in (1, 2, 3):
        phrase = _get_trailing_phrase(text, n)
        if phrase and phrase in TAGALOG_SOFT_DANGLERS:
            return True
    return False


def _ends_with_dangling(text: str, char_len: int = 0) -> bool:
    """
    Return True if the segment should be merged due to a dangling ending.
    Hard danglers always trigger. Soft danglers only trigger when char_len < SOFT_DANGLING_MAX_CHARS.
    """
    if _ends_with_hard_dangler(text):
        return True
    if char_len < SOFT_DANGLING_MAX_CHARS and _ends_with_soft_dangler(text):
        return True
    return False


def _do_merge(cur, nxt):
    """Merge nxt segment into cur in-place (words + timing)."""
    if cur.words and nxt.words:
        last_word = cur.words[-1]
        if not last_word.word.endswith(' '):
            last_word.word = last_word.word + ' '
        cur.words.extend(nxt.words)
    cur.end = nxt.end


# Minimum character length for a segment to be considered "long enough to stand alone"
# even without terminal punctuation. Prevents merging well-formed subtitle cards
# that happen to not end with a period (common in conversational Taglish/English).
MERGE_STANDALONE_MIN_CHARS = 55

def merge_broken_sentences(result, max_merge_gap: float = 0.15):
    """
    Merge adjacent segments when the current segment is a genuinely incomplete
    phrase — i.e. Whisper split mid-clause with a near-zero gap.

    Merge triggers:
      (A) cur ends WITHOUT terminal punctuation AND gap <= max_merge_gap (0.15s)
          AND cur is short enough that it clearly needs continuation (< 55 chars)
      (B) cur ends with a Tagalog dangling word/phrase (pa rin, wala pa, etc.)
          AND cur is short (< 55 chars) — long segments with dangling words are
          usually complete subtitle cards, not broken sentences.

    Hard blocks (never merge even if A or B is true):
      - cur text is >= MERGE_STANDALONE_MIN_CHARS — already a complete card
      - nxt starts a NEW SENTENCE (capital letter) AND cur ends with .!?
      - gap > 1.5s
      - merged result would exceed 140 chars
    """
    segs = result.segments
    i = 0
    while i < len(segs) - 1:
        cur      = segs[i]
        nxt      = segs[i + 1]
        cur_text = _seg_text(cur)
        nxt_text = _seg_text(nxt)

        gap        = nxt.start - cur.end
        merged_len = len(cur_text) + 1 + len(nxt_text)

        # Hard block: too long, too far apart, or would exceed limit
        if merged_len > 140 or gap > 1.5:
            i += 1
            continue

        # Hard block: cur is already long enough to be a standalone card
        if len(cur_text) >= MERGE_STANDALONE_MIN_CHARS:
            i += 1
            continue

        # Hard block: sentence boundary — nxt is a new sentence
        nxt_is_new_sentence = _starts_new_sentence(nxt_text)
        cur_ends_sentence   = _prev_ends_sentence(cur_text)
        if nxt_is_new_sentence and cur_ends_sentence:
            i += 1
            continue

        # Condition A: cur ends mid-phrase and gap is very small (Whisper cut mid-clause)
        # Two sub-cases:
        #   A1 — very short fragment (< 25 chars) with no terminal punct → always merge
        #   A2 — ends with a COMMA (syntactically incomplete regardless of length) → merge
        #        Comma-ending segments like "...sa Tondo," always need their continuation.
        no_terminal = cur_text and cur_text[-1] not in SENTENCE_END_CHARS
        ends_with_comma = cur_text.endswith(',')
        # Condition A fires only when:
        #   - ends with a comma (syntactically always incomplete), OR
        #   - segment is truly bare (≤ 10 chars, e.g. "ng", "sa akin") — hard cut by Whisper
        # Short segments ending with non-comma non-dangling words (like "as we prepare yung")
        # are valid Taglish subtitle cards and should stay as-is.
        cond_a = (no_terminal and gap <= max_merge_gap and
                  (ends_with_comma or len(cur_text) <= 10))

        # Condition B: cur ends with a Tagalog dangling particle/phrase
        cond_b = _ends_with_dangling(cur_text, len(cur_text)) and not (nxt_is_new_sentence and cur_ends_sentence)

        if cond_a or cond_b:
            _do_merge(cur, nxt)
            segs.pop(i + 1)
        else:
            i += 1

    return result


# ─────────────────────────────────────────────
#  SMART SEGMENTATION ENGINE  (v2)
# ─────────────────────────────────────────────
#
#  New philosophy:
#    Whisper transcribes with word_timestamps=True and regroup=False.
#    We get a flat word timeline with real inter-word gaps.
#    This engine is the ONLY thing that decides where segments start/end.
#
#  Two-pass algorithm:
#    PASS 1 — find all candidate split points scored by quality:
#      Each inter-word gap is scored. Higher score = better split point.
#      Score factors:
#        + large gap (silence)           → strongest signal
#        + sentence-ending punctuation   → strong signal
#        + comma                         → medium signal
#        - splitting a proper noun pair  → penalty
#        - leaving a very short remainder→ penalty
#
#    PASS 2 — greedily build segments:
#      Walk the word list. At each candidate split point, commit if:
#        - buffer has reached IDEAL length (42+ chars), OR
#        - buffer has reached MAX length (84 chars) — force split
#        - buffer duration exceeds 7s    — force split
#      Never split below MIN_SEGMENT_CHARS unless forced.
# ──────────────────────────────────────────────

SEG_MAX_CHARS         = 84    # hard max chars per segment (2 lines × 42)
SEG_IDEAL_CHARS       = 42    # start considering splits at this length
SEG_MIN_CHARS         = 12    # never produce a segment shorter than this
SEG_MAX_DURATION      = 7.0   # Netflix hard max seconds


def _collect_words(result):
    """Flatten all WordTiming objects into a single list, skip blanks."""
    words = []
    for seg in result.segments:
        if seg.words:
            for w in seg.words:
                if w.word.strip():
                    words.append(w)
    return words


def _gap_score(words, i):
    """
    Score the split point AFTER words[i] (before words[i+1]).
    Higher = better split point.
    Returns 0.0 if i is the last word.
    """
    if i >= len(words) - 1:
        return 0.0

    cur  = words[i]
    nxt  = words[i + 1]
    gap  = nxt.start - cur.end
    wt   = cur.word.strip()
    score = 0.0

    # ── Gap size (primary signal) ──────────────────────────────────
    if   gap >= 2.0:  score += 100.0
    elif gap >= 1.0:  score +=  60.0
    elif gap >= 0.5:  score +=  30.0
    elif gap >= 0.3:  score +=  15.0
    elif gap >= 0.15: score +=   5.0
    # gaps < 0.15s get no gap bonus — continuous speech

    # ── Punctuation bonuses (stacked on top of gap) ────────────────
    if wt and wt[-1] in '.!?':
        score += 25.0          # sentence end = very good split
    elif wt and wt[-1] == ',':
        score +=  8.0          # comma = decent split

    # ── Penalties ─────────────────────────────────────────────────
    # Penalise splitting between two Title-Case words (proper noun pair)
    nwt = nxt.word.strip()
    if wt and wt[0].isupper() and nwt and nwt[0].isupper():
        score -= 20.0

    return score


def smart_segment(result, max_chars=None, max_duration=None):
    """
    Rebuild subtitle segments purely from word-level timestamps.
    Returns list of dicts: [{'start', 'end', 'text'}]
    """
    words = _collect_words(result)
    if not words:
        return []

    eff_max  = int(max_chars)      if max_chars      else SEG_MAX_CHARS
    eff_dur  = float(max_duration) if max_duration   else SEG_MAX_DURATION

    # ── Pre-compute gap scores for every inter-word boundary ──────
    scores = [_gap_score(words, i) for i in range(len(words))]

    # ── Greedy segment builder ─────────────────────────────────────
    segments_out = []
    buf_start    = 0   # index of first word in current buffer

    def commit(end_idx_inclusive):
        """Flush words[buf_start..end_idx_inclusive] as one segment."""
        wlist = words[buf_start: end_idx_inclusive + 1]
        text  = ' '.join(w.word.strip() for w in wlist)
        if text.strip():
            segments_out.append({
                'start': wlist[0].start,
                'end':   wlist[-1].end,
                'text':  text.strip(),
            })

    i = buf_start
    while i < len(words):
        is_last     = (i == len(words) - 1)
        buf_words   = words[buf_start: i + 1]
        buf_text    = ' '.join(w.word.strip() for w in buf_words)
        buf_chars   = len(buf_text)
        buf_dur     = buf_words[-1].end - buf_words[0].start if len(buf_words) > 1 else 0

        # ── FORCE SPLIT conditions (override everything) ───────────
        force = is_last or buf_chars >= eff_max or buf_dur >= eff_dur

        if force:
            commit(i)
            buf_start = i + 1
            i = buf_start
            continue

        # ── CANDIDATE SPLIT: only consider if buffer is long enough ─
        if buf_chars >= SEG_IDEAL_CHARS and scores[i] > 0:
            # Look ahead: would the remainder be too short?
            remaining_words = words[i + 1:]
            remaining_chars = sum(len(w.word.strip()) + 1 for w in remaining_words)
            if remaining_chars < SEG_MIN_CHARS:
                # Absorb remainder into this segment
                commit(len(words) - 1)
                buf_start = len(words)
                break

            commit(i)
            buf_start = i + 1
            i = buf_start
            continue

        # ── OPPORTUNISTIC SPLIT: split on strong signal regardless of length ─
        # Score >= 60 means gap >= 1s — clear speaker change or long pause.
        # We always split here even if the buffer is short, to prevent
        # two different speakers being merged into one subtitle card.
        # Score >= 40 (gap >= 0.5s) only splits if buffer has content.
        opp_threshold = 60.0 if buf_chars < SEG_MIN_CHARS else 40.0
        if scores[i] >= opp_threshold:
            remaining_words = words[i + 1:]
            remaining_chars = sum(len(w.word.strip()) + 1 for w in remaining_words)
            if remaining_chars >= SEG_MIN_CHARS:
                commit(i)
                buf_start = i + 1
                i = buf_start
                continue

        i += 1

    # Flush any remainder
    if buf_start < len(words):
        commit(len(words) - 1)

    return segments_out


# ─────────────────────────────────────────────
#  MAIN PROCESS FUNCTION

# ─────────────────────────────────────────────

def process_media(
    model_size, source_lang, upload, model_type,
    max_chars, max_words, extend_in, extend_out, collapse_gaps,
    max_lines_per_segment, line_penalty, longest_line_char_penalty,
    initial_prompt,
    # new params
    vad_threshold, vad_min_silence_ms, gap_threshold,
    beam_size, use_netflix_style, enable_missed_recovery,
    enable_hallucination_filter, enable_punctuation_fix,
    *args
):
    # ── resolve initial prompt ──────────────────
    if not initial_prompt or not initial_prompt.strip():
        initial_prompt = DEFAULT_TAGLISH_PROMPT if source_lang == "tl" else None

    start_time = time.time()

    if upload is None:
        return None, None, None, None, "No file uploaded."

    original_path = upload.name

    # ── 1. PRE-PROCESS: FFmpeg normalize ────────
    temp_path = preprocess_audio(original_path)

    # ── 2. LOAD MODEL ───────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # VAD parameters
    vad_params = dict(
        threshold=float(vad_threshold),
        min_speech_duration_ms=150,
        min_silence_duration_ms=int(vad_min_silence_ms),
        speech_pad_ms=250,
    )

    bs = int(beam_size)

    if model_type == "faster whisper":
        compute = "float16" if device == "cuda" else "int8"
        model = stable_whisper.load_faster_whisper(
            model_size, device=device, compute_type=compute
        )
        result = model.transcribe(
            temp_path,
            language=source_lang,
            vad=True,
            vad_parameters=vad_params,
            regroup=False,
            #denoiser="demucs",               # ← was commented out — now active
            beam_size=bs,
            best_of=bs,
            patience=1.5,
            no_speech_threshold=0.9,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
        )
    else:
        model = stable_whisper.load_model(model_size, device=device)
        result = model.transcribe(
            temp_path,
            language=source_lang,
            vad=True,
            regroup=False,
            no_speech_threshold=0.9,
            #denoiser="demucs",
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
        )

    # ── 3. MISSED SEGMENT RECOVERY ──────────────
    missed_report = ""
    if enable_missed_recovery:
        audio_duration = get_audio_duration(temp_path)
        missed = find_missed_segments(result, audio_duration, min_gap=float(gap_threshold))
        if missed:
            recovered = retry_missed_segments(model, temp_path, missed, source_lang, model_type)
            missed_report = (
                f"🔍 Found {len(missed)} gap(s). Recovered {len(recovered)} segment(s)."
            )
            # Merge recovered segments back into result (stable-ts supports list-like access)
            for seg in recovered:
                result.segments.append(seg)
            result.segments.sort(key=lambda s: s.start)
        else:
            missed_report = "✅ No significant gaps detected."

    # ── 4 & 5. HALLUCINATION FILTER + PUNCTUATION FIX ──────────
    # NOTE: stable-ts Segment.text is READ-ONLY — we must patch word tokens.
    if enable_hallucination_filter or (enable_punctuation_fix and source_lang == "tl"):
        apply_text_filters_to_result(
            result,
            enable_hallucination_filter=enable_hallucination_filter,
            enable_punctuation_fix=enable_punctuation_fix,
            source_lang=source_lang,
        )

    # ── 6. SMART SEGMENTATION ───────────────────
    # Rebuild segments from word-level timestamps using linguistic rules.
    # Replaces all merge/split hacks with a single clean algorithm.
    smart_segs = smart_segment(
        result,
        max_chars=int(max_chars) if max_chars else None,
        max_duration=NETFLIX_MAX_DURATION,
    )

    # ── 8. ANTI-FLICKER ─────────────────────────
    extend_start      = float(extend_in)    if extend_in    else 0.0
    extend_end        = float(extend_out)   if extend_out   else 0.0
    collapse_gaps_val = float(collapse_gaps) if collapse_gaps else 0.0

    for i in range(len(smart_segs) - 1):
        cur = smart_segs[i]
        nxt = smart_segs[i + 1]
        if nxt['start'] - cur['end'] < extend_start + extend_end:
            k = extend_end / (extend_start + extend_end) if (extend_start + extend_end) > 0 else 0
            mid = cur['end'] * (1 - k) + nxt['start'] * k
            cur['end'] = nxt['start'] = mid
        else:
            cur['end']   += extend_end
            nxt['start'] -= extend_start
            if nxt['start'] - cur['end'] <= collapse_gaps_val:
                cur['end'] = nxt['start'] = (cur['end'] + nxt['start']) / 2

    if smart_segs:
        smart_segs[0]['start'] = max(0, smart_segs[0]['start'] - extend_start)
        smart_segs[-1]['end']  += extend_end

    # ── 9. NETFLIX STYLE ────────────────────────
    netflix_report = ""
    if use_netflix_style:
        netflix_report = check_netflix_style(
            smart_segs,
            max_lines_per_segment=int(max_lines_per_segment) if max_lines_per_segment else 2,
            line_penalty=float(line_penalty) if line_penalty else 22.01,
            longest_line_char_penalty=float(longest_line_char_penalty) if longest_line_char_penalty else 1.0,
        )

    # ── 10. SRT OUTPUT ──────────────────────────
    lps  = int(max_lines_per_segment)      if max_lines_per_segment      else 2
    lpen = float(line_penalty)             if line_penalty               else 22.01
    lcpen= float(longest_line_char_penalty)if longest_line_char_penalty  else 1.0

    original_filename = os.path.splitext(os.path.basename(original_path))[0]
    srt_dir           = tempfile.gettempdir()
    subtitles_path    = os.path.join(srt_dir, f"{original_filename}.srt")

    srt_blocks = []
    for idx, seg in enumerate(smart_segs):
        srt_blocks.append(segment2optimizedsrtblock(seg, idx, lps, lpen, lcpen))
    srt_content = '\n\n'.join(srt_blocks)

    with open(subtitles_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)

    transcript_txt = ' '.join(s['text'] for s in smart_segs)

    elapsed = time.time() - start_time

    # Build status report
    seg_chars = [len(s['text']) for s in smart_segs]
    avg_chars = sum(seg_chars) / len(seg_chars) if seg_chars else 0
    long_segs = sum(1 for c in seg_chars if c > NETFLIX_MAX_CPL)
    status_parts = [
        f"⏱️ Processed in {elapsed:.1f}s  |  Device: {device.upper()}  |  v2.0",
        f"📊 {len(smart_segs)} segments  |  avg {avg_chars:.0f} chars  |  {long_segs} segments split into 2 lines"
    ]
    if missed_report:
        status_parts.append(missed_report)
    if netflix_report:
        status_parts.append(netflix_report)
    status = "\n".join(status_parts)

    # Clean up temp preprocessed file
    if temp_path != original_path:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

    mime, _ = mimetypes.guess_type(original_path)
    audio_out = original_path if mime and mime.startswith("audio") else None
    video_out = original_path if mime and mime.startswith("video") else None

    return audio_out, video_out, transcript_txt, subtitles_path, status


# ─────────────────────────────────────────────
#  YOUTUBE / UTILITY FUNCTIONS  (unchanged)
# ─────────────────────────────────────────────

def extract_playlist_to_csv(playlist_url, cookies_path=None):
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'dump_single_json': True
    }
    try:
        cookies_path = _normalize_file_path(cookies_path)
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
                    title    = video.get('title', 'N/A')
                    video_id = video['id']
                    url      = f'https://www.youtube.com/watch?v={video_id}'
                    writer.writerow([title, video_id, url])
        return csv_path
    except Exception:
        return None


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

        downloads_dir   = os.path.join(os.path.expanduser("~"), "Downloads")
        output_template = os.path.join(downloads_dir, "%(id)s.%(ext)s")

        errors       = []
        cookies_path = _normalize_file_path(cookies_path)
        try:
            if shutil.which("yt-dlp"):
                for url in urls:
                    if not url:
                        continue
                    cmd = [
                        "yt-dlp",
                        "--write-subs", "--write-auto-subs",
                        "--sub-lang", "en-US",
                        "--skip-download",
                        "--convert-subs", "srt",
                        "-o", output_template,
                        url
                    ]
                    if cookies_path:
                        cmd.extend(["--cookies", cookies_path])
                    try:
                        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
                        print(r.stdout); print(r.stderr)
                    except Exception as e:
                        errors.append(f"{url}: {e}")
            else:
                ydl_opts = {
                    'writesubtitles': True, 'writeautomaticsub': True,
                    'subtitleslangs': ['en-US', 'en'],
                    'skip_download': True, 'outtmpl': output_template,
                    'quiet': True, 'subtitlesformat': 'srt'
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

        temp_dir     = tempfile.mkdtemp(prefix="ssui_srt_")
        copied_paths = []
        copy_errors  = []
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


def _normalize_file_path(file_input):
    if not file_input:
        return None
    if isinstance(file_input, str):
        return file_input
    if isinstance(file_input, dict):
        for k in ("name", "tmp_path", "tempfile", "file_path", "path"):
            if k in file_input and file_input[k]:
                return file_input[k]
        return None
    try:
        return getattr(file_input, "name", None)
    except Exception:
        return None


def check_youtube_tag(video_url, tag_to_check, cookies_path=None):
    try:
        cookies_path = _normalize_file_path(cookies_path)
        ydl_opts = {"quiet": True}
        if cookies_path:
            ydl_opts["cookies"] = cookies_path
        ydl_opts.setdefault("http_headers", {})
        ydl_opts["http_headers"].setdefault(
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        with YoutubeDL(ydl_opts) as ydl:
            info           = ydl.extract_info(video_url, download=False)
            tags           = info.get('tags', [])
            tag_norm       = tag_to_check.lower()
            tags_norm      = [t.lower() for t in tags]
            exists         = any(tag_norm == t for t in tags_norm)
            if exists:
                return f"Tag/s '{tag_to_check}' EXISTS in video"
            return f"Tag/s '{tag_to_check}' DOES NOT EXIST in video.\n\nTags found: {tags or 'None'}"
    except Exception as e:
        err = str(e)
        if 'Sign in to confirm your age' in err or ('Sign in' in err and 'age' in err):
            return f"Error checking {video_url}: Age-restricted — provide a cookies.txt file."
        if 'HTTP Error 403' in err or '403' in err:
            return f"Error checking {video_url}: HTTP 403 Forbidden — try a cookies file or update yt-dlp."
        return f"Error checking {video_url}: {err}"


def check_playlist_tags(playlist_url, tag_to_check, cookies_path=None):
    try:
        cookies_path = _normalize_file_path(cookies_path)
        ydl_opts = {'extract_flat': True, 'quiet': True, 'dump_single_json': True}
        if cookies_path:
            ydl_opts['cookies'] = cookies_path
        ydl_opts.setdefault("http_headers", {})
        ydl_opts["http_headers"].setdefault(
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        with YoutubeDL(ydl_opts) as ydl:
            result         = ydl.extract_info(playlist_url, download=False)
            entries        = result.get('entries', [])
            rows           = []
            tag_norm       = tag_to_check.lower()
            for video in entries:
                video_id = video.get('id')
                if not video_id:
                    rows.append([video.get('title', 'N/A'), '', 'No video ID in playlist entry'])
                    continue
                video_url = f'https://www.youtube.com/watch?v={video_id}'
                title     = video.get('title', 'N/A')
                video_opts = {'quiet': True}
                if cookies_path:
                    video_opts['cookies'] = cookies_path
                video_opts.setdefault("http_headers", {})
                video_opts["http_headers"].setdefault(
                    "User-Agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                try:
                    with YoutubeDL(video_opts) as ydl_video:
                        info       = ydl_video.extract_info(video_url, download=False)
                        is_unlisted = info.get('is_unlisted', False) if isinstance(info, dict) else False
                        is_private  = info.get('is_private', False)  if isinstance(info, dict) else False
                        age_limit   = info.get('age_limit', 0)       if isinstance(info, dict) else 0
                        tags        = info.get('tags', []) or []
                        tags_norm   = [t.lower() for t in tags]
                        exists      = any(tag_norm == t for t in tags_norm)
                        parts = []
                        if is_unlisted: parts.append('Unlisted')
                        if is_private:  parts.append('Private')
                        elif age_limit and int(age_limit) >= 18: parts.append('Age-restricted')
                        parts.append(f"Tag '{tag_to_check}' {'exists' if exists else 'does NOT exist'}")
                        rows.append([title, video_url, '; '.join(parts)])
                except Exception as e:
                    err       = str(e)
                    err_lower = err.lower()
                    if 'sign in to confirm your age' in err_lower:
                        note = 'Age-restricted - cookies required'
                    elif 'private' in err_lower and 'video' in err_lower:
                        note = 'Private video - access denied'
                    elif 'video unavailable' in err_lower or 'not available' in err_lower:
                        note = 'Video unavailable or removed'
                    elif '403' in err_lower or 'forbidden' in err_lower:
                        note = 'HTTP 403 Forbidden - cookies may be required'
                    else:
                        note = f"Could not check video: {err}"
                    rows.append([title, video_url, note])

        fd, csv_path = tempfile.mkstemp(suffix=".csv", text=True)
        os.close(fd)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Title", "URL", "Notes"])
            writer.writerows(rows)
        return csv_path
    except Exception as e:
        fd, csv_path = tempfile.mkstemp(suffix=".csv", text=True)
        os.close(fd)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Title", "URL", "Notes"])
            writer.writerow(["Error", "", str(e)])
        return csv_path


# ─────────────────────────────────────────────
#  LANGUAGE LIST
# ─────────────────────────────────────────────

WHISPER_LANGUAGES = [
    ("Afrikaans", "af"), ("Albanian", "sq"), ("Amharic", "am"),
    ("Arabic", "ar"), ("Armenian", "hy"), ("Assamese", "as"),
    ("Azerbaijani", "az"), ("Bashkir", "ba"), ("Basque", "eu"),
    ("Belarusian", "be"), ("Bengali", "bn"), ("Bosnian", "bs"),
    ("Breton", "br"), ("Bulgarian", "bg"), ("Burmese", "my"),
    ("Catalan", "ca"), ("Chinese", "zh"), ("Croatian", "hr"),
    ("Czech", "cs"), ("Danish", "da"), ("Dutch", "nl"),
    ("English", "en"), ("Estonian", "et"), ("Faroese", "fo"),
    ("Finnish", "fi"), ("French", "fr"), ("Galician", "gl"),
    ("Georgian", "ka"), ("German", "de"), ("Greek", "el"),
    ("Gujarati", "gu"), ("Haitian Creole", "ht"), ("Hausa", "ha"),
    ("Hebrew", "he"), ("Hindi", "hi"), ("Hungarian", "hu"),
    ("Icelandic", "is"), ("Indonesian", "id"), ("Italian", "it"),
    ("Japanese", "ja"), ("Javanese", "jv"), ("Kannada", "kn"),
    ("Kazakh", "kk"), ("Khmer", "km"), ("Korean", "ko"),
    ("Lao", "lo"), ("Latin", "la"), ("Latvian", "lv"),
    ("Lingala", "ln"), ("Lithuanian", "lt"), ("Luxembourgish", "lb"),
    ("Macedonian", "mk"), ("Malagasy", "mg"), ("Malay", "ms"),
    ("Malayalam", "ml"), ("Maltese", "mt"), ("Maori", "mi"),
    ("Marathi", "mr"), ("Mongolian", "mn"), ("Nepali", "ne"),
    ("Norwegian", "no"), ("Nyanja", "ny"), ("Occitan", "oc"),
    ("Pashto", "ps"), ("Persian", "fa"), ("Polish", "pl"),
    ("Portuguese", "pt"), ("Punjabi", "pa"), ("Romanian", "ro"),
    ("Russian", "ru"), ("Sanskrit", "sa"), ("Serbian", "sr"),
    ("Shona", "sn"), ("Sindhi", "sd"), ("Sinhala", "si"),
    ("Slovak", "sk"), ("Slovenian", "sl"), ("Somali", "so"),
    ("Spanish", "es"), ("Sundanese", "su"), ("Swahili", "sw"),
    ("Swedish", "sv"), ("Tagalog", "tl"), ("Tajik", "tg"),
    ("Tamil", "ta"), ("Tatar", "tt"), ("Telugu", "te"),
    ("Thai", "th"), ("Turkish", "tr"), ("Turkmen", "tk"),
    ("Ukrainian", "uk"), ("Urdu", "ur"), ("Uzbek", "uz"),
    ("Vietnamese", "vi"), ("Welsh", "cy"), ("Yiddish", "yi"),
    ("Yoruba", "yo"),
]


# ─────────────────────────────────────────────
#  GRADIO UI
# ─────────────────────────────────────────────

with gr.Blocks() as interface:
    gr.HTML(
        """
        <style>.html-container.svelte-phx28p.padding { padding: 0 !important; }</style>
        <div class='custom-container'>
        <h1 style='text-align: left;'>Speech Solutions✨</h1>
        <p style='text-align: left;'> Hosted on 🤗
            <a href="https://huggingface.co/spaces/DeeeeeM/ssui-app" target="_blank">
                <b>Hugging Face Spaces</b>
            </a>
        </p>
        """
    )
    gr.Markdown(
        """
        AI-powered speech processing with Whisper + stable-ts.

        - Speech-to-text (WhisperAI) with Taglish optimizations
        - Netflix-style subtitle formatting
        - Missed speech recovery
        - Language translation (GPT-4) *(In progress)*
        - Text to Speech *(In progress)*

        UPDATE: Now includes YouTube metadata extraction features.
        """
    )

    with gr.Tabs():
        # ── SPEECH TO TEXT TAB ──────────────────────────────────────────────
        with gr.TabItem("Speech to Text"):
            gr.HTML("<h2 style='text-align: left;'>OpenAI Whisper + stable-ts</h2>")
            gr.Markdown(
                """
                Whisper speech recognition with stable-ts accurate timestamps,
                audio pre-processing, hallucination filtering, Netflix-style output,
                and Taglish/Filipino optimizations.
                """
            )

            # ── General Settings ────────────────────────────────────────────
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Upload Audio or Video",
                        file_types=["audio", "video"]
                    )

                with gr.Column(scale=1):
                    with gr.Group():
                        source_lang = gr.Dropdown(
                            choices=WHISPER_LANGUAGES,
                            label="Source Language",
                            value="tl",
                            interactive=True
                        )
                        model_type = gr.Dropdown(
                            choices=["faster whisper", "whisper"],
                            label="Model Type",
                            value="faster whisper",
                            interactive=True
                        )
                        model_size = gr.Dropdown(
                            choices=[
                                "deepdml/faster-whisper-large-v3-turbo-ct2",
                                "large-v3-turbo",
                                "large-v3",
                                "large-v2",
                                "large",
                                "medium",
                                "small",
                                "base",
                                "tiny"
                            ],
                            label="Model Size",
                            value="large-v3-turbo",   # ← changed default for speed
                            interactive=True
                        )
                        initial_prompt = gr.Textbox(
                            label="Initial Prompt (optional — auto-filled for Tagalog)",
                            lines=3,
                            placeholder="Leave blank to use the built-in Taglish prompt for Filipino content.",
                            interactive=True
                        )

            # ── Quality & Speed Settings ─────────────────────────────────────
            with gr.Accordion("⚙️ Quality & Speed Settings", open=False):
                gr.Markdown(
                    """
                    **Beam Size** controls accuracy vs speed. Higher = more accurate, slower.
                    **VAD** controls how aggressively silence is filtered before transcription.
                    """
                )
                with gr.Row():
                    with gr.Column():
                        beam_size = gr.Slider(
                            minimum=1, maximum=15, value=5, step=1,
                            label="Beam Size",
                            info="Higher = more accurate, slower. 5 is balanced, 10 is high accuracy."
                        )
                    with gr.Column():
                        vad_threshold = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.40, step=0.05,
                            label="VAD Sensitivity Threshold",
                            info="Lower = picks up more speech. 0.40 is good for Filipino pacing."
                        )
                    with gr.Column():
                        vad_min_silence_ms = gr.Slider(
                            minimum=100, maximum=1000, value=350, step=50,
                            label="VAD Min Silence (ms)",
                            info="Minimum silence duration to split segments. 350ms suits conversational Filipino."
                        )

            # ── Post-Processing Settings ─────────────────────────────────────
            with gr.Accordion("🧹 Post-Processing Settings", open=False):
                with gr.Row():
                    with gr.Column():
                        enable_hallucination_filter = gr.Checkbox(
                            label="🚫 Hallucination Filter",
                            value=True,
                            info="Remove repeated phrases and known Whisper hallucinations."
                        )
                        enable_punctuation_fix = gr.Checkbox(
                            label="✏️ Tagalog Punctuation Fix",
                            value=True,
                            info="Add commas after Tagalog discourse markers (kasi, naman, diba, etc.)."
                        )
                    with gr.Column():
                        use_netflix_style = gr.Checkbox(
                            label="🎬 Netflix Style Guide",
                            value=False,
                            info="Enforce CPS ≤17, CPL ≤42, max 2 lines, duration 0.2–7s."
                        )

            # ── Missed Speech Recovery ──────────────────────────────────────
            with gr.Accordion("🔍 Missed Speech Recovery", open=False):
                gr.Markdown(
                    """
                    Detects gaps in the transcription and re-runs Whisper on those segments
                    with more sensitive settings to recover missed speech.
                    """
                )
                with gr.Row():
                    with gr.Column():
                        enable_missed_recovery = gr.Checkbox(
                            label="Enable Missed Speech Recovery",
                            value=False,
                            info="Re-process gaps longer than the threshold below. Adds processing time."
                        )
                    with gr.Column():
                        gap_threshold = gr.Slider(
                            minimum=0.5, maximum=10.0, value=2.0, step=0.5,
                            label="Gap Detection Threshold (seconds)",
                            info="Gaps longer than this will be re-processed."
                        )

            # ── Advanced Subtitle Settings ───────────────────────────────────
            with gr.Accordion("📐 Advanced Subtitle Settings", open=False):
                gr.Markdown(
                    """
                    Control how subtitle segments are split and formatted.
                    Default values follow Netflix subtitle guidelines (max 2 lines, 42 CPL).

                    <i><b>Note: Changing these may override Netflix Style Guide settings above.</b></i>
                    """
                )
                with gr.Row():
                    with gr.Column():
                        max_chars = gr.Number(
                            label="Max Chars Per Segment",
                            info="Whisper outputs up to this many chars per segment (1 line). Smart splitting then breaks long segments into 2 lines at 42 chars.",
                            value=84, precision=0, interactive=True
                        )
                        max_words = gr.Number(
                            label="Max Words (disabled)",
                            info="Leave blank — smart splitting handles line breaks automatically.",
                            value=None, precision=0, interactive=True
                        )
                        max_lines_per_segment = gr.Number(
                            label="Max Lines Per Segment",
                            info="Smart splitting will produce up to this many lines. Netflix standard is 2.",
                            value=2, precision=0, interactive=True
                        )
                    with gr.Column():
                        extend_in = gr.Number(
                            label="Extend In (s)",
                            info="Extend segment start by this value",
                            value=0, precision=2
                        )
                        extend_out = gr.Number(
                            label="Extend Out (s)",
                            info="Extend segment end by this value",
                            value=0.5, precision=2, interactive=True
                        )
                        collapse_gaps = gr.Number(
                            label="Collapse Gaps (s)",
                            info="Collapse gaps between segments shorter than this",
                            value=0.3, precision=2, interactive=True
                        )
                    with gr.Column():
                        line_penalty = gr.Number(
                            label="Line Penalty",
                            info="Penalty per additional line when optimizing layout",
                            value=22.01, precision=2, interactive=True
                        )
                        longest_line_char_penalty = gr.Number(
                            label="Longest Line Char Penalty",
                            info="Penalty per character of the longest line",
                            value=1, precision=2, interactive=True
                        )

            submit_btn = gr.Button("▶ PROCESS", variant="primary")

            with gr.Row():
                with gr.Column():
                    status_output = gr.Textbox(
                        label="Processing Status",
                        lines=4,
                        interactive=False
                    )
                    transcript_output = gr.Textbox(
                        label="Transcript",
                        lines=8,
                        interactive=False
                    )
                    srt_output = gr.File(label="Download SRT", interactive=False)
                with gr.Column():
                    video_output = gr.Video(label="Video Output")
                    audio_output = gr.Audio(label="Audio Output")

            submit_btn.click(
                fn=process_media,
                inputs=[
                    model_size, source_lang, file_input, model_type,
                    max_chars, max_words, extend_in, extend_out, collapse_gaps,
                    max_lines_per_segment, line_penalty, longest_line_char_penalty,
                    initial_prompt,
                    vad_threshold, vad_min_silence_ms, gap_threshold,
                    beam_size, use_netflix_style, enable_missed_recovery,
                    enable_hallucination_filter, enable_punctuation_fix,
                ],
                outputs=[audio_output, video_output, transcript_output, srt_output, status_output]
            )

        # ── YOUTUBE PLAYLIST EXTRACTOR ──────────────────────────────────────
        with gr.TabItem("Youtube playlist extractor"):
            gr.Markdown("### Extract YT Title, URL, and ID from a YouTube playlist and download as CSV.")
            playlist_url = gr.Textbox(label="YouTube Playlist URL", placeholder="Paste playlist URL here")
            cookie_file_extract = gr.File(label="YouTube Cookies File (optional)", file_types=None, interactive=True)
            process_btn = gr.Button("Process")
            csv_output  = gr.File(label="Download CSV")
            process_btn.click(
                extract_playlist_to_csv,
                inputs=[playlist_url, cookie_file_extract],
                outputs=csv_output
            )

        # ── SRT DOWNLOADER ──────────────────────────────────────────────────
        with gr.TabItem("SRT Downloader"):
            gr.Markdown("### Download English subtitles (.srt) from YouTube video(s). Separate URLs with commas or newlines.")
            srt_url         = gr.Textbox(label="YouTube Video URL", placeholder="Paste video URL here")
            cookie_file_srt = gr.File(label="YouTube Cookies File (optional)", file_types=None, interactive=True)
            srt_btn         = gr.Button("Process")
            srt_file        = gr.File(label="Download SRT")
            srt_status      = gr.Textbox(label="Status", interactive=False)
            srt_btn.click(
                download_srt,
                inputs=[srt_url, cookie_file_srt],
                outputs=[srt_file, srt_status]
            )

        # ── TAG CHECKER ─────────────────────────────────────────────────────
        with gr.TabItem("Tag Checker"):
            gr.Markdown("### Check if a specific tag exists in a YouTube video's metadata.")
            gr.Markdown("*Tip: If a video is age-restricted, export cookies from your browser and upload below.*")
            tag_url     = gr.Textbox(label="YouTube Video URL", placeholder="Paste video URL here")
            tag_input   = gr.Textbox(label="Tag to Check", placeholder="e.g. series:my father's wife")
            cookie_file_tag = gr.File(label="YouTube Cookies File (optional)", file_types=None, interactive=True)
            tag_btn     = gr.Button("Process")
            tag_output  = gr.Textbox(label="Tag Check Result", interactive=False)
            tag_btn.click(
                check_youtube_tag,
                inputs=[tag_url, tag_input, cookie_file_tag],
                outputs=tag_output
            )

        # ── PLAYLIST TAG CHECKER ────────────────────────────────────────────
        with gr.TabItem("Playlist Tag Checker"):
            gr.Markdown(
                """
                Check if a specific tag exists in all videos of a YouTube playlist.

                <b><i>Note: This may take longer depending on playlist size.</i></b>
                """
            )
            gr.Markdown("*Tip: Upload a cookies.txt file for age-restricted videos.*")
            playlist_url_tags   = gr.Textbox(label="YouTube Playlist URL", placeholder="Paste playlist URL here")
            tag_input_playlist  = gr.Textbox(label="Tag to Check", placeholder="e.g. series:my father's wife")
            cookie_file_playlist = gr.File(label="YouTube Cookies File (optional)", file_types=None, interactive=True)
            tag_btn_playlist    = gr.Button("Process")
            tag_output_playlist = gr.File(label="Download Tag Check CSV", interactive=False)
            tag_btn_playlist.click(
                check_playlist_tags,
                inputs=[playlist_url_tags, tag_input_playlist, cookie_file_playlist],
                outputs=tag_output_playlist
            )

    gr.HTML(
        """
        <audio id="notify-audio" src="https://www.soundjay.com/buttons/sounds/button-3.mp3"></audio>
        <script>
        function playNotify() {
            var audio = document.getElementById('notify-audio');
            if (audio) { audio.play(); }
        }
        document.addEventListener('DOMContentLoaded', function() {
            let outputs = document.querySelectorAll("textarea, input[type='file'], video, audio");
            outputs.forEach(function(output) {
                output.addEventListener("change", playNotify);
            });
        });
        </script>
        """
    )

interface.launch(share=True)