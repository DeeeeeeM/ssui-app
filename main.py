import gradio as gr 
import mimetypes
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import stable_whisper
from stable_whisper.text_output import result_to_any, sec2srt
import tempfile
import re
import textwrap
import torch

# --- Main function to process the media file --- #
def process_media(
    model_size, source_lang, upload, model_type,
    max_chars, max_words, extend_in, extend_out, collapse_gaps,
    max_lines_per_segment, line_penalty, longest_line_char_penalty, *args
):
    # ----- is file empty? checker ----- #
    if upload is None:
        return None, None, None, None 

    temp_path = upload.name
    base_path = os.path.splitext(temp_path)[0]
    word_transcription_path = base_path + '.json'

    # ---- Load .json or transcribe ---- #
    if os.path.exists(word_transcription_path):
        print(f"Transcription data file found at {word_transcription_path}")
        result = stable_whisper.WhisperResult(word_transcription_path)
    else:
        print(f"Can't find transcription data file at {word_transcription_path}. Starting transcribing ...")

        #-- Check if CUDA is available or not --#
        if model_type == "faster whisper":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = stable_whisper.load_faster_whisper(model_size, device=device)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = stable_whisper.load_model(model_size, device=device)

        try:
            result = model.transcribe(temp_path, language=source_lang, vad=True, regroup=False, denoiser="demucs")
        except Exception as e:
            return None, None, None, None  # Remove the 5th value
        result.save_as_json(word_transcription_path)

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

    return audio_out, video_out, transcript_txt, srt_file_path  # Only 4 values

def optimize_text(text, max_lines_per_segment, line_penalty, longest_line_char_penalty):
    text = text.strip()
    words = text.split()

    psum = [0]
    for w in words:
        psum += [psum[-1] + len(w) + 1]  

    bestScore = 10 ** 30
    bestSplit = None

    def backtrack(level, wordsUsed, maxLineLength, split):
        nonlocal bestScore, bestSplit

        if wordsUsed == len(words):
            score = level * line_penalty + maxLineLength * longest_line_char_penalty
            if score < bestScore:
                bestScore = score
                bestSplit = split
            return

        if level + 1 == max_lines_per_segment:
            backtrack(
                level + 1, len(words),
                max(maxLineLength, psum[len(words)] - psum[wordsUsed] - 1),
                split + [words[wordsUsed:]]
            )
            return

        for levelWords in range(1, len(words) - wordsUsed + 1):
            backtrack(
                level + 1, wordsUsed + levelWords,
                max(maxLineLength, psum[wordsUsed + levelWords] - psum[wordsUsed] - 1),
                split + [words[wordsUsed:wordsUsed + levelWords]]
            )

    backtrack(0, 0, 0, [])

    optimized = '\n'.join(' '.join(words) for words in bestSplit)
    return optimized

def segment2optimizedsrtblock(segment: dict, idx: int, max_lines_per_segment, line_penalty, longest_line_char_penalty, strip=True) -> str:
    return f'{idx}\n{sec2srt(segment["start"])} --> {sec2srt(segment["end"])}\n' \
           f'{optimize_text(segment["text"], max_lines_per_segment, line_penalty, longest_line_char_penalty)}'

def segments2blocks(segments, max_lines_per_segment, line_penalty, longest_line_char_penalty):
    return '\n\n'.join(
        segment2optimizedsrtblock(s, i, max_lines_per_segment, line_penalty, longest_line_char_penalty, strip=True)
        for i, s in enumerate(segments)
    )

WHISPER_LANGUAGES = [
    ("Afrikaans", "af"),
    ("Albanian", "sq"),
    ("Amharic", "am"),
    ("Arabic", "ar"),
    ("Armenian", "hy"),
    ("Assamese", "as"),
    ("Azerbaijani", "az"),
    ("Bashkir", "ba"),
    ("Basque", "eu"),
    ("Belarusian", "be"),
    ("Bengali", "bn"),
    ("Bosnian", "bs"),
    ("Breton", "br"),
    ("Bulgarian", "bg"),
    ("Burmese", "my"),
    ("Catalan", "ca"),
    ("Chinese", "zh"),
    ("Croatian", "hr"),
    ("Czech", "cs"),
    ("Danish", "da"),
    ("Dutch", "nl"),
    ("English", "en"),
    ("Estonian", "et"),
    ("Faroese", "fo"),
    ("Finnish", "fi"),
    ("French", "fr"),
    ("Galician", "gl"),
    ("Georgian", "ka"),
    ("German", "de"),
    ("Greek", "el"),
    ("Gujarati", "gu"),
    ("Haitian Creole", "ht"),
    ("Hausa", "ha"),
    ("Hebrew", "he"),
    ("Hindi", "hi"),
    ("Hungarian", "hu"),
    ("Icelandic", "is"),
    ("Indonesian", "id"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Javanese", "jv"),
    ("Kannada", "kn"),
    ("Kazakh", "kk"),
    ("Khmer", "km"),
    ("Korean", "ko"),
    ("Lao", "lo"),
    ("Latin", "la"),
    ("Latvian", "lv"),
    ("Lingala", "ln"),
    ("Lithuanian", "lt"),
    ("Luxembourgish", "lb"),
    ("Macedonian", "mk"),
    ("Malagasy", "mg"),
    ("Malay", "ms"),
    ("Malayalam", "ml"),
    ("Maltese", "mt"),
    ("Maori", "mi"),
    ("Marathi", "mr"),
    ("Mongolian", "mn"),
    ("Nepali", "ne"),
    ("Norwegian", "no"),
    ("Nyanja", "ny"),
    ("Occitan", "oc"),
    ("Pashto", "ps"),
    ("Persian", "fa"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Punjabi", "pa"),
    ("Romanian", "ro"),
    ("Russian", "ru"),
    ("Sanskrit", "sa"),
    ("Serbian", "sr"),
    ("Shona", "sn"),
    ("Sindhi", "sd"),
    ("Sinhala", "si"),
    ("Slovak", "sk"),
    ("Slovenian", "sl"),
    ("Somali", "so"),
    ("Spanish", "es"),
    ("Sundanese", "su"),
    ("Swahili", "sw"),
    ("Swedish", "sv"),
    ("Tagalog", "tl"),
    ("Tajik", "tg"),
    ("Tamil", "ta"),
    ("Tatar", "tt"),
    ("Telugu", "te"),
    ("Thai", "th"),
    ("Turkish", "tr"),
    ("Turkmen", "tk"),
    ("Ukrainian", "uk"),
    ("Urdu", "ur"),
    ("Uzbek", "uz"),
    ("Vietnamese", "vi"),
    ("Welsh", "cy"),
    ("Yiddish", "yi"),
    ("Yoruba", "yo"),
]

with gr.Blocks() as interface:
    gr.HTML(
        """
        <style>.html-container.svelte-phx28p.padding { padding: 0 !important; }</style>
        <div class='custom-container'>
        <h1 style='text-align: left;'>Speech Solutions</h1>
        """
    )
    gr.Markdown(
    """
    This is a simple Gradio UI app that combines AI-powered speech and language processing technologies. This app supports the following features:

    - Speech-to-text (WhisperAI)
    - Language translation (GPT-4) (In progress)

    <b>NOTE: This app is currently in the process of applying other AI-solutions for other use cases.</b>
    """
    )

    with gr.Tabs():
        with gr.TabItem("Speech to Text"):
            gr.HTML("<h2 style='text-align: left;'>OpenAI/Whisper + stable-ts</h1>")
            gr.Markdown(
            """ 
            Open Ai's <b>Whisper</b> is a versatile speech recognition model trained on diverse audio for tasks like multilingual transcription, translation, and language ID. With the help of <b>stable-ts</b>, it provides accurate word-level timestamps in chronological order without extra processing.
            """
            )
            #General Settings
            with gr.Row():
                #Media Input
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload Audio or Video", file_types=["audio", "video"])
                #Settings
                with gr.Column(scale=1):
                    with gr.Group():
                        source_lang = gr.Dropdown(
                            choices=WHISPER_LANGUAGES,
                            label="Source Language",
                            value="tl",  # default to Tagalog
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
                                ("Large v3 Turbo", "large-v3-turbo"),
                                ("Large v3", "large-v3"),
                                ("Large v2", "large-v2"),
                                ("Large", "large"),
                                ("Medium", "medium"),
                                ("Small", "small"),
                                ("Base", "base"),
                                ("Tiny", "tiny")
                            ],
                            label="Model Size",
                            value="large-v2",
                            interactive=True
                        )
            #Advanced Settings
            with gr.Accordion("Advanced Settings", open=False):
                gr.Markdown(
                    """ 
                    These settings allow you to customize the segmentation of the audio or video file. Adjust these parameters to control how the segments are created based on characters, words, and lines.

                    <b><i>Note: The values currently set are the default values. You can adjust them to your needs, but be aware that changing these values may affect the segmentation of the audio or video file.</i></b>
                    """
                )
                with gr.Row():
                    with gr.Column():
                        max_chars = gr.Number(
                            label="Max Chars",
                            info="Maximum characters allowed in segment",
                            value=86,
                            precision=0,
                            interactive=True
                        )
                        max_words = gr.Number(
                            label="Max Words",
                            info="Maximum words allowed in segment",
                            value=30,
                            precision=0,
                            interactive=True
                        )
                        max_lines_per_segment = gr.Number(
                            label="Max Lines Per Segment",
                            info="Max lines allowed per subtitle segment",
                            value=3,
                            precision=0,
                            interactive=True
                        )
                    with gr.Column():
                        extend_in = gr.Number(
                            label="Extend In",
                            info="Extend the start of all segments by this value (in seconds)",
                            value=0,
                            precision=2,
                            
                        )
                        extend_out = gr.Number(
                            label="Extend Out",
                            info="Extend the end of all segments by this value (in seconds)",
                            value=0.5,
                            precision=2,
                            interactive=True
                        )
                        collapse_gaps = gr.Number(
                            label="Collapse Gaps",
                            info="Collapse gaps between segments under a certain duration",
                            value=0.3,
                            precision=2,
                            interactive=True
                        )
                        
                    with gr.Column():
                        line_penalty = gr.Number(
                            label="Longest Line Character",
                            info="Penalty for each additional line (used to decide when to split segment into several lines)",
                            value=22.01,
                            precision=2,
                            interactive=True
                        )
                        longest_line_char_penalty = gr.Number(
                            label="Longest Line Character",
                            info="Penalty for each character of the longest segment line (used to decide when to split segment into several lines)",
                            value=1,
                            precision=2,
                            interactive=True
                        )
            submit_btn = gr.Button("PROCESS", elem_id="orange-process-btn")            
            with gr.Row(): 
                with gr.Column():
                    transcript_output = gr.Textbox(label="Transcript", lines=8, interactive=False)
                    srt_output = gr.File(label="Download SRT", interactive=False)

                with gr.Column():
                    video_output = gr.Video(label="Video Output")
                    audio_output = gr.Audio(label="Audio Output")

            submit_btn.click(
                fn=process_media,
                inputs=[
                    model_size, source_lang, file_input, model_type,
                    max_chars, max_words, extend_in, extend_out, collapse_gaps,
                    max_lines_per_segment, line_penalty, longest_line_char_penalty
                ],
                outputs=[audio_output, video_output, transcript_output, srt_output]
            )

        with gr.TabItem("..."):
            pass

interface.launch(share=True)