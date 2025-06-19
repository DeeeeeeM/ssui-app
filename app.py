import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tempfile
import mimetypes
import gradio as gr 
import torch
import stable_whisper
from stable_whisper.text_output import result_to_any, sec2srt
import time
from yt_dlp import YoutubeDL
import csv
import os

def process_media(
    model_size, source_lang, upload, model_type,
    max_chars, max_words, extend_in, extend_out, collapse_gaps,
    max_lines_per_segment, line_penalty, longest_line_char_penalty,
    initial_prompt=None,  #
    *args
):
    if not initial_prompt:
        initial_prompt = None 

    start_time = time.time()

    if upload is None:
        return None, None, None, None 

    temp_path = upload.name

    #-- Check if CUDA is available or not --#
    if model_type == "faster whisper":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = stable_whisper.load_faster_whisper(model_size, device=device)
        result = model.transcribe(
            temp_path,
            language=source_lang,
            vad=True,
            regroup=False,
            no_speech_threshold=0.9,
            initial_prompt=initial_prompt  # <-- pass here
        )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = stable_whisper.load_model(model_size, device=device)
        result = model.transcribe(
            temp_path,
            language=source_lang,
            vad=True,
            regroup=False,
            no_speech_threshold=0.9,
            denoiser="demucs",
            initial_prompt=initial_prompt  # <-- pass here
        )
    #, batch_size=16, denoiser="demucs"
    #result.save_as_json(word_transcription_path) 

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

    elapsed = time.time() - start_time 
    print(f"process_media completed in {elapsed:.2f} seconds")

    return audio_out, video_out, transcript_txt, srt_file_path

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

    if not bestSplit:
        return text
        
    if len(bestSplit) > max_lines_per_segment or any(len(line) == 1 for line in bestSplit):
        return text

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

def extract_playlist_to_csv(playlist_url):
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'dump_single_json': True
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(playlist_url, download=False)
            entries = result.get('entries', [])
            # Save to a temp file for download
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
    This is a Gradio UI app that combines AI-powered speech and language processing technologies. This app supports the following features:

    - Speech-to-text (WhisperAI)
    - Language translation (GPT-4) (In progress)
    - Improved transcription (GPT-4) (In progress)
    - Text to Speech (In progress)

    <i><b>NOTE: This app is currently in the process of applying other AI-solutions for other use cases.</b></i>
    """
    )

    with gr.Tabs():
        with gr.TabItem("Speech to Text"):
            gr.HTML("<h2 style='text-align: left;'>OpenAI / Whisper + stable-ts</h2>")
            gr.Markdown(
            """ 
            Open Ai's <b>Whisper</b> is a versatile speech recognition model trained on diverse audio for tasks like multilingual transcription, translation, and language ID. With the help of <b>stable-ts</b>, it provides accurate word-level timestamps in chronological order without extra processing.

            <i>Note: The default values are set for balanced and faster processing, 
            you can choose: large, large v2, and large v3 <b>MODEL SIZE</b> for more accuracy, but they may take longer to process.</i>

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
                                "large-v3",
                                "large-v2",
                                "large",
                                "medium",
                                "small",
                                "base",
                                "tiny"
                            ],
                            label="Model Size",
                            value="deepdml/faster-whisper-large-v3-turbo-ct2",
                            interactive=True
                        )
                        initial_prompt = gr.Textbox(
                            label="Initial Prompt (optional)",
                            lines=3,
                            placeholder="Add context, names, or style for the model here",
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
            submit_btn = gr.Button("- PROCESS -")            
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

        with gr.TabItem("Youtube playlist extractor"):
            gr.Markdown("### Extract YT Title, URL, and ID from a YouTube playlist and download as CSV.")
            playlist_url = gr.Textbox(label="YouTube Playlist URL", placeholder="Paste playlist URL here")
            process_btn = gr.Button("Process")
            csv_output = gr.File(label="Download CSV")
            process_btn.click(
                extract_playlist_to_csv,
                inputs=playlist_url,
                outputs=csv_output
            )


interface.launch(share=True)