from .sub_optimize import segments2blocks

import time
import os
import tempfile
import mimetypes

import torch
import stable_whisper
from stable_whisper.text_output import result_to_any

def process_media(
    model_size, source_lang, upload, model_type,
    max_chars, max_words, extend_in, extend_out, collapse_gaps,
    max_lines_per_segment, line_penalty, longest_line_char_penalty,
    initial_prompt=None, *args
):
    start_time = time.time()
    
    initial_prompt = initial_prompt if initial_prompt else None

    if upload is None:
        return None, None, None, None 

    temp_path = upload.name

    if model_type == "faster whisper":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = stable_whisper.load_faster_whisper(model_size, device=device)
        result = model.transcribe(
            temp_path,
            language=source_lang,
            vad=True,
            regroup=False,
            #batch_size=16,
            initial_prompt=initial_prompt)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = stable_whisper.load_model(model_size, device=device)
        result = model.transcribe(
            temp_path,
            language=source_lang,
            vad=True,
            regroup=False,
            #no_speech_threshold=0.9,
            initial_prompt=initial_prompt
        )

    # ADVANCED SETTINGS #
    if max_chars or max_words:
        result.split_by_length(
            max_chars=int(max_chars) if max_chars else None,
            max_words=int(max_words) if max_words else None
        )

    # ----- Anti-flickering ----- #
    extend_start = float(extend_in) if extend_in else 0.0
    extend_end = float(extend_out) if extend_out else 0.0
    collapse_gaps_under = float(collapse_gaps) if collapse_gaps else 0.0

    for i in range(len(result) - 1):
        cur = result[i]
        next = result[i+1]

        if next.start - cur.end < extend_start + extend_end:
            k = extend_end / (extend_start + extend_end) if (extend_start + extend_end) > 0 else 0
            mid = cur.end * (1 - k) + next.start * k
            cur.end = next.start = mid
        else:
            cur.end += extend_end
            next.start -= extend_start

            if next.start - cur.end <= collapse_gaps_under:
                cur.end = next.start = (cur.end + next.start) / 2

    if result:
        result[0].start = max(0, result[0].start - extend_start)
        result[-1].end += extend_end

    # --- Custom SRT block output --- #
    original_filename = os.path.splitext(os.path.basename(temp_path))[0]
    srt_dir = tempfile.gettempdir()
    subtitles_path = os.path.join(srt_dir, f"{original_filename}.srt")

    result_to_any(
        result=result,
        filepath=subtitles_path,
        filetype='srt',
        segments2blocks=lambda segments: segments2blocks(
            segments,
            int(max_lines_per_segment) if max_lines_per_segment else 3,
            float(line_penalty) if line_penalty else 22.01,
            float(longest_line_char_penalty) if longest_line_char_penalty else 1.0
        ),
        word_level=False,
    )
    srt_file_path = subtitles_path
    transcript_txt = result.to_txt()

    mime, _ = mimetypes.guess_type(temp_path)
    audio_out = temp_path if mime and mime.startswith("audio") else None
    video_out = temp_path if mime and mime.startswith("video") else None


    return audio_out, video_out, transcript_txt, srt_file_path