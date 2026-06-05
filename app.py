from modules.languages import WHISPER_LANGUAGES
from modules.playlist_extractor import extract_playlist_to_csv
from modules.tag_checker import check_playlist_tags, check_youtube_tag
from modules.sub_dl import download_srt
from modules.main import process_media
from modules.srt_translator import translate_srt
from modules.video_downloader import download_single_video, download_playlist
from modules.speech_input_helpers import _process_local_single, _process_local_multiple, _process_youtube_single, _process_youtube_playlist
from modules.yt_filter_wrapper import filter_youtube_urls
from modules.utils import normalize_file_path
import os
import glob
import tempfile
import zipfile

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import gradio as gr

LANGUAGE_DEFAULT_PROMPTS = {
    "tl": "Ang sumusunod ay isang pag-uusap sa Filipino o Tagalog. Maaaring may halong Ingles na salita.",
    "fil": "Ang sumusunod ay isang pag-uusap sa Filipino o Tagalog. Maaaring may halong Ingles na salita.",
}

def update_prompt_for_language(lang):
    return LANGUAGE_DEFAULT_PROMPTS.get(lang, "")

# ── YouTube Utilities Wrappers ──────────────────────────────────────────

def extract_playlist_with_status(playlist_url, cookies_path=None):
    """Wrapper for playlist extraction with status message."""
    if not playlist_url or not playlist_url.strip():
        return None, "❌ Error: Please provide a playlist URL"
    
    try:
        result = extract_playlist_to_csv(playlist_url, cookies_path)
        if result:
            return result, "✅ Playlist extracted successfully"
        else:
            return None, "❌ Error: Could not extract playlist. Check URL or cookies."
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def download_srt_with_feedback(input_mode, video_urls, cookies_path=None):
    """Wrapper for SRT download with improved feedback."""
    if not video_urls or not str(video_urls).strip():
        return None, "❌ Error: Please provide at least one YouTube URL"
    
    try:
        result, message = download_srt(video_urls, cookies_path)
        if input_mode == "Playlist":
            if result and isinstance(result, str) and result.lower().endswith('.zip'):
                message = "✅ Playlist subtitles downloaded and archived"
            else:
                message = "✅ Playlist subtitles download completed"
        return result, message
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def check_youtube_tag_with_feedback(video_url, tag_to_check, cookies_path=None):
    """Wrapper for tag checker with error handling."""
    if not video_url or not video_url.strip():
        return "❌ Error: Please provide a video URL"
    
    if not tag_to_check or not tag_to_check.strip():
        return "❌ Error: Please provide a tag to check"
    
    try:
        result = check_youtube_tag(video_url, tag_to_check, cookies_path)
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"

def check_playlist_tags_with_feedback(playlist_url, tag_to_check, cookies_path=None):
    """Wrapper for playlist tag checker with error handling."""
    if not playlist_url or not playlist_url.strip():
        return None, "❌ Error: Please provide a playlist URL"
    
    if not tag_to_check or not tag_to_check.strip():
        return None, "❌ Error: Please provide a tag to check"
    
    try:
        result = check_playlist_tags(playlist_url, tag_to_check, cookies_path)
        if result:
            return result, "✅ Playlist tag check complete"
        else:
            return None, "❌ Error: Could not check playlist tags"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def process_with_status(model_size, source_lang, input_mode,
                        local_single_file, local_multiple_files,
                        youtube_video_url, youtube_playlist_url, youtube_cookies,
                        model_type, max_chars, max_words, extend_in, extend_out, collapse_gaps,
                        max_lines_per_segment, line_penalty, longest_line_char_penalty,
                        initial_prompt, normalize_audio, use_demucs,
                        enable_translation, target_lang, service, api_key, ollama_host):
    yield gr.update(value="⏳ Processing... this may take a few minutes."), None, None, None, None
    try:
        if input_mode == "Local Single":
            status, audio_out, video_out, transcript, srt = _process_local_single(
                local_single_file, model_size, source_lang, model_type,
                max_chars, max_words, extend_in, extend_out, collapse_gaps,
                max_lines_per_segment, line_penalty, longest_line_char_penalty,
                initial_prompt, normalize_audio, use_demucs,
                enable_translation, target_lang, service, api_key, ollama_host
            )
        elif input_mode == "Local Multiple":
            status, audio_out, video_out, transcript, srt = _process_local_multiple(
                local_multiple_files, model_size, source_lang, model_type,
                max_chars, max_words, extend_in, extend_out, collapse_gaps,
                max_lines_per_segment, line_penalty, longest_line_char_penalty,
                initial_prompt, normalize_audio, use_demucs,
                enable_translation, target_lang, service, api_key, ollama_host
            )
        elif input_mode == "YouTube Single":
            status, audio_out, video_out, transcript, srt = _process_youtube_single(
                youtube_video_url, youtube_cookies, model_size, source_lang, model_type,
                max_chars, max_words, extend_in, extend_out, collapse_gaps,
                max_lines_per_segment, line_penalty, longest_line_char_penalty,
                initial_prompt, normalize_audio, use_demucs,
                enable_translation, target_lang, service, api_key, ollama_host
            )
        else:
            status, audio_out, video_out, transcript, srt = _process_youtube_playlist(
                youtube_playlist_url, youtube_cookies, model_size, source_lang, model_type,
                max_chars, max_words, extend_in, extend_out, collapse_gaps,
                max_lines_per_segment, line_penalty, longest_line_char_penalty,
                initial_prompt, normalize_audio, use_demucs,
                enable_translation, target_lang, service, api_key, ollama_host
            )

        if enable_translation and srt is not None and not str(srt).lower().endswith(".zip"):
            try:
                translated_srt_path, trans_status = translate_srt(
                    srt_file_path=srt,
                    target_lang=target_lang,
                    service=service,
                    api_key=api_key if api_key else None,
                    ollama_host=ollama_host
                )
                if translated_srt_path:
                    srt = translated_srt_path
                    yield gr.update(value=f"✅ Done! {status} {trans_status}"), audio_out, video_out, transcript, srt
                else:
                    yield gr.update(value=f"⚠️ {status} Translated failed: {trans_status}"), audio_out, video_out, transcript, srt
            except Exception as e:
                yield gr.update(value=f"⚠️ {status} Transcribed successfully but translation error: {str(e)}"), audio_out, video_out, transcript, srt
        else:
            if enable_translation and srt is not None and str(srt).lower().endswith(".zip"):
                status = f"{status} ⚠️ Translation is not available for multiple-file ZIP outputs."
            yield gr.update(value=status), audio_out, video_out, transcript, srt
    except Exception as e:
        yield gr.update(value=f"❌ Error: {str(e)}"), None, None, None, None


CSS = """
.process-btn {
    background: #2563eb !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin-top: 8px !important;
}
.process-btn:hover {
    background: #1d4ed8 !important;
}
.status-box textarea {
    font-size: 0.9rem;
    color: #374151;
    background: #f9fafb;
    border-radius: 6px;
}
.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
"""

with gr.Blocks(css=CSS, title="NMI Media Toolkit") as interface:

    gr.HTML("""
        <div style="padding: 16px 0 8px 0;">
            <h1 style="margin: 0; font-size: 1.6rem;">NMI Media Toolkit ✨</h1>
            <p style="margin: 4px 0 0 0; color: #6b7280; font-size: 0.9rem;">
                Hosted on 🤗
                <a href="https://huggingface.co/spaces/DeeeeeM/ssui-app" target="_blank"><b>Hugging Face Spaces</b></a>
            </p>
        </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Speech to Text ──────────────────────────────────────────
        with gr.TabItem("🎙️ Speech to Text"):

            with gr.Row(equal_height=False):

                # ── LEFT: Inputs ───────────────────────────────────────────
                with gr.Column(scale=1):

                    gr.HTML('<p class="section-label">Input</p>')
                    input_mode = gr.Radio(
                        choices=[
                            "Local Single",
                            "Local Multiple",
                            "YouTube Single",
                            "YouTube Playlist"
                        ],
                        label="Input Mode",
                        value="Local Single",
                        interactive=True
                    )
                    local_single_file = gr.File(
                        label="Upload Audio or Video",
                        file_types=["audio", "video"]
                    )
                    local_multiple_files = gr.File(
                        label="Upload Multiple Audio or Video Files",
                        file_types=["audio", "video"],
                        file_count="multiple",
                        visible=False
                    )
                    youtube_video_url = gr.Textbox(
                        label="YouTube Video URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        visible=False
                    )
                    youtube_playlist_url = gr.Textbox(
                        label="YouTube Playlist URL",
                        placeholder="https://www.youtube.com/playlist?list=...",
                        visible=False
                    )
                    youtube_cookies = gr.File(
                        label="YouTube Cookies (optional)",
                        file_types=["text"],
                        visible=False
                    )

                    gr.HTML('<p class="section-label">Translation Options</p>')
                    gr.Markdown("Translate generated SRT to another language after transcription.")
                    
                    enable_translation = gr.Checkbox(
                        label="Enable Subtitle Translation",
                        value=False,
                        interactive=True
                    )
                    
                    with gr.Row(visible=False) as translation_row:
                        with gr.Column():
                            target_lang = gr.Dropdown(
                                choices=[
                                    ("English", "en"),
                                    ("Spanish", "es"),
                                    ("French", "fr"),
                                    ("German", "de"),
                                    ("Italian", "it"),
                                    ("Portuguese", "pt"),
                                    ("Russian", "ru"),
                                    ("Japanese", "ja"),
                                    ("Korean", "ko"),
                                    ("Chinese", "zh"),
                                    ("Arabic", "ar"),
                                    ("Turkish", "tr"),
                                    ("Filipino/Tagalog", "tl"),
                                    ("Vietnamese", "vi"),
                                    ("Thai", "th"),
                                    ("Indonesian", "id"),
                                ],
                                label="Target Language",
                                value="en",
                                interactive=True
                            )
                        
                        with gr.Column():
                            service = gr.Dropdown(
                                choices=[
                                    ("Google Translate (Free)", "google"),
                                    ("DeepL (Free Tier)", "deepl"),
                                    ("OpenAI GPT", "openai"),
                                    ("Deepseek (Free Tier)", "deepseek"),
                                    ("Local LLM / Ollama", "local_llm"),
                                ],
                                label="Translation Service",
                                value="google",
                                interactive=True
                            )
                        
                        with gr.Column():
                            api_key = gr.Textbox(
                                label="API Key (if needed)",
                                placeholder="Leave blank for env variable or Google/Local LLM",
                                type="password",
                                interactive=True
                            )
                    
                    with gr.Row(visible=False) as ollama_row:
                        ollama_host = gr.Textbox(
                            label="Ollama Host",
                            value="",
                            placeholder="http://localhost:11434",
                            interactive=True
                        )
                    
                    # Toggle translation options visibility
                    def toggle_translation(enable):
                        return gr.update(visible=enable), gr.update(visible=enable)
                    
                    enable_translation.change(
                        fn=toggle_translation,
                        inputs=enable_translation,
                        outputs=[translation_row, ollama_row]
                    )

                    gr.HTML('<p class="section-label">Model</p>')
                    with gr.Row():
                        model_type = gr.Dropdown(
                            choices=["faster whisper", "whisper"],
                            label="Transcription Engine",
                            value="faster whisper",
                            interactive=True
                        )
                        model_size = gr.Dropdown(
                            choices=["large-v3-turbo", "large-v3", "large-v2", "large", "medium", "small", "base", "tiny"],
                            label="Model",
                            value="large-v3",
                            interactive=True
                        )

                    source_lang = gr.Dropdown(
                        choices=WHISPER_LANGUAGES,
                        label="Source Language",
                        value="tl",
                        interactive=True
                    )

                    initial_prompt = gr.Textbox(
                        label="Initial Prompt",
                        lines=2,
                        value=LANGUAGE_DEFAULT_PROMPTS.get("tl", ""),
                        placeholder="Leave blank for no prompt, or customize for your content.",
                        interactive=True
                    )

                    gr.HTML('<p class="section-label">Audio Preprocessing</p>')
                    with gr.Row():
                        normalize_audio = gr.Checkbox(
                            label="Normalize Audio",
                            info="Fixes inconsistent loudness (recommended)",
                            value=True
                        )
                        use_demucs = gr.Checkbox(
                            label="Vocal Separation",
                            info="Removes BG music via demucs (slower)",
                            value=False
                        )

                    with gr.Accordion("🎛️ Preprocessing Options", open=False):
                        gr.Markdown("Fine-tune subtitle segmentation. Default values work well for most content.")
                        with gr.Row():
                            with gr.Column():
                                max_chars = gr.Number(
                                    label="Max Chars Per Line",
                                    info="Max characters per subtitle line",
                                    value=84, precision=0, interactive=True
                                )
                                max_words = gr.Number(
                                    label="Max Words",
                                    info="Max words per segment",
                                    value=30, precision=0, interactive=True
                                )
                                max_lines_per_segment = gr.Number(
                                    label="Max Lines Per Segment",
                                    value=3, precision=0, interactive=True
                                )
                            with gr.Column():
                                extend_in = gr.Number(
                                    label="Extend In (s)",
                                    info="Extend segment start",
                                    value=0, precision=2
                                )
                                extend_out = gr.Number(
                                    label="Extend Out (s)",
                                    info="Extend segment end",
                                    value=0.5, precision=2, interactive=True
                                )
                                collapse_gaps = gr.Number(
                                    label="Collapse Gaps (s)",
                                    info="Merge gaps shorter than this",
                                    value=0.3, precision=2, interactive=True
                                )
                            with gr.Column():
                                line_penalty = gr.Number(
                                    label="Line Penalty",
                                    info="Penalty per extra line when splitting",
                                    value=22.01, precision=2, interactive=True
                                )
                                longest_line_char_penalty = gr.Number(
                                    label="Char Penalty (Longest Line)",
                                    info="Penalty per character of the longest line",
                                    value=1, precision=2, interactive=True
                                )

                    def toggle_speech_inputs(selected):
                        return (
                            gr.update(visible=selected == "Local Single"),
                            gr.update(visible=selected == "Local Multiple"),
                            gr.update(visible=selected == "YouTube Single"),
                            gr.update(visible=selected == "YouTube Playlist"),
                            gr.update(visible=selected.startswith("YouTube"))
                        )

                    input_mode.change(
                        fn=toggle_speech_inputs,
                        inputs=input_mode,
                        outputs=[
                            local_single_file,
                            local_multiple_files,
                            youtube_video_url,
                            youtube_playlist_url,
                            youtube_cookies,
                        ]
                    )

                    status_box = gr.Textbox(
                        label="Status",
                        value="Ready.",
                        interactive=False,
                        elem_classes=["status-box"]
                    )

                    submit_btn = gr.Button("Process", elem_classes=["process-btn"])

                # ── RIGHT: Outputs ──────────────────────────────────────────
                with gr.Column(scale=1):

                    gr.HTML('<p class="section-label">Output</p>')
                    transcript_output = gr.Textbox(
                        label="Transcript",
                        lines=10,
                        interactive=False
                    )
                    srt_output = gr.File(label="Download SRT", interactive=False)
                    video_output = gr.Video(label="Video Preview")
                    audio_output = gr.Audio(label="Audio Preview")

            # Auto-update prompt when language changes
            source_lang.change(
                fn=update_prompt_for_language,
                inputs=source_lang,
                outputs=initial_prompt
            )

            submit_btn.click(
                fn=process_with_status,
                inputs=[
                    model_size, source_lang, input_mode, local_single_file, local_multiple_files,
                    youtube_video_url, youtube_playlist_url, youtube_cookies, model_type,
                    max_chars, max_words, extend_in, extend_out, collapse_gaps,
                    max_lines_per_segment, line_penalty, longest_line_char_penalty,
                    initial_prompt, normalize_audio, use_demucs,
                    enable_translation, target_lang, service, api_key, ollama_host
                ],
                outputs=[status_box, audio_output, video_output, transcript_output, srt_output]
            )

        # ── Tab 2: YouTube Playlist Extractor ─────────────────────────────
        with gr.TabItem("📋 Playlist Extractor"):
            gr.Markdown("Extract title, URL, and ID from a YouTube playlist as a CSV.")
            with gr.Row():
                with gr.Column():
                    playlist_url = gr.Textbox(label="YouTube Playlist URL", placeholder="Paste playlist URL here")
                    cookie_file_extract = gr.File(label="Cookies File (optional)")
                    process_btn = gr.Button("Extract", elem_classes=["process-btn"])
                with gr.Column():
                    extract_status = gr.Textbox(label="Status", value="Ready.", interactive=False, elem_classes=["status-box"])
                    csv_output = gr.File(label="Download CSV")
            
            def extract_handler(url, cookies):
                csv_path, status = extract_playlist_with_status(url, cookies)
                return status, csv_path
            
            process_btn.click(
                extract_handler,
                inputs=[playlist_url, cookie_file_extract],
                outputs=[extract_status, csv_output]
            )

        # ── Tab 3: SRT Downloader ─────────────────────────────────────────
        with gr.TabItem("📥 SRT Downloader"):
            gr.Markdown("Download subtitles (.srt) from YouTube videos or playlists. Separate multiple video URLs with commas.")
            srt_mode = gr.Radio(
                choices=["Video(s)", "Playlist"],
                value="Video(s)",
                label="Input Type",
                interactive=True
            )
            with gr.Row():
                with gr.Column():
                    srt_url = gr.Textbox(label="YouTube URL(s)", placeholder="Paste a video URL, playlist URL, or multiple video URLs here")
                    cookie_file_srt = gr.File(label="Cookies File (optional)")
                    srt_btn = gr.Button("Download", elem_classes=["process-btn"])
                with gr.Column():
                    srt_status = gr.Textbox(label="Status", value="Ready.", interactive=False, elem_classes=["status-box"])
                    srt_file = gr.File(label="Download SRT")
            
            def download_handler(mode, urls, cookies):
                srt_path, status = download_srt_with_feedback(mode, urls, cookies)
                return status, srt_path
            
            srt_btn.click(
                download_handler,
                inputs=[srt_mode, srt_url, cookie_file_srt],
                outputs=[srt_status, srt_file]
            )

        # ── Tab 4: Video Downloader (MP3/MP4) ─────────────────────────────
        with gr.TabItem("⬇️ Video Downloader"):
            gr.Markdown(
                "Download YouTube videos or audio in MP3/MP4 format.\n\n"
                "Choose your preferred format and quality. Single videos save to Downloads folder, playlists create a subfolder."
            )
            
            # Mode selector
            dl_mode = gr.Radio(
                choices=["Single Video", "Playlist"],
                value="Single Video",
                label="Download Mode",
                interactive=True
            )
            
            with gr.Row():
                with gr.Column():
                    # Single Video Input
                    with gr.Row(visible=True) as dl_single_row:
                        dl_video_url = gr.Textbox(
                            label="YouTube Video URL",
                            placeholder="Paste video URL here",
                            interactive=True
                        )
                    
                    # Playlist Input
                    with gr.Row(visible=False) as dl_playlist_row:
                        dl_playlist_url = gr.Textbox(
                            label="YouTube Playlist URL",
                            placeholder="Paste playlist URL here",
                            interactive=True
                        )
                    
                    # Format and quality
                    with gr.Row():
                        dl_format = gr.Dropdown(
                            choices=["MP3", "MP4"],
                            value="MP4",
                            label="Format",
                            interactive=True
                        )
                        dl_quality = gr.Dropdown(
                            choices=["High", "Medium", "Low"],
                            value="Medium",
                            label="Quality",
                            interactive=True
                        )
                    
                    dl_cookie_file = gr.File(label="Cookies File (optional)")
                    dl_btn = gr.Button("Download", elem_classes=["process-btn"])
                
                with gr.Column():
                    dl_status = gr.Textbox(
                        label="Status",
                        value="Ready.",
                        interactive=False,
                        elem_classes=["status-box"]
                    )
                    
                    # Single video output
                    with gr.Row(visible=True) as dl_single_output_row:
                        dl_file = gr.File(label="Download File")
                    
                    # Playlist output
                    with gr.Row(visible=False) as dl_playlist_output_row:
                        gr.Markdown("Playlist files are saved to:\n`~/Downloads/playlist_download/`\n\nCheck your Downloads folder to access downloaded files.")
            
            # Toggle visibility based on mode
            def toggle_dl_mode(mode):
                if mode == "Single Video":
                    return (
                        gr.update(visible=True),   # dl_single_row
                        gr.update(visible=False),  # dl_playlist_row
                        gr.update(visible=True),   # dl_single_output_row
                        gr.update(visible=False)   # dl_playlist_output_row
                    )
                else:
                    return (
                        gr.update(visible=False),  # dl_single_row
                        gr.update(visible=True),   # dl_playlist_row
                        gr.update(visible=False),  # dl_single_output_row
                        gr.update(visible=True)    # dl_playlist_output_row
                    )
            
            dl_mode.change(
                fn=toggle_dl_mode,
                inputs=dl_mode,
                outputs=[dl_single_row, dl_playlist_row, dl_single_output_row, dl_playlist_output_row]
            )
            
            # Handler function
            def download_video_handler(mode, video_url, playlist_url, format_type, quality, cookies):
                format_lower = format_type.lower()
                quality_lower = quality.lower()
                
                if mode == "Single Video":
                    file_path, status = download_single_video(video_url, format_lower, quality_lower, cookies)
                    return status, file_path
                else:
                    folder_path, status = download_playlist(playlist_url, format_lower, quality_lower, cookies)
                    return status, None
            
            dl_btn.click(
                fn=download_video_handler,
                inputs=[dl_mode, dl_video_url, dl_playlist_url, dl_format, dl_quality, dl_cookie_file],
                outputs=[dl_status, dl_file]
            )

        # ── Tab 5: Tag Checker (Single Video & Playlist) ──────────────────
        with gr.TabItem("🏷️ Tag Checker"):
            gr.Markdown(
                "Check if a specific tag exists in YouTube videos.\n\n"
                "*If videos are age-restricted, export cookies from your browser using the 'Get cookies.txt' extension and upload below.*"
            )
            
            # Mode selector
            tag_mode = gr.Radio(
                choices=["Single Video", "Playlist"],
                value="Single Video",
                label="Check Mode",
                interactive=True
            )
            
            with gr.Row():
                with gr.Column():
                    # Single Video Input
                    with gr.Row(visible=True) as single_video_row:
                        tag_url = gr.Textbox(
                            label="YouTube Video URL",
                            placeholder="Paste video URL here",
                            interactive=True
                        )
                    
                    # Playlist Input
                    with gr.Row(visible=False) as playlist_row:
                        playlist_url_tags = gr.Textbox(
                            label="YouTube Playlist URL",
                            placeholder="Paste playlist URL here",
                            interactive=True
                        )
                    
                    tag_input = gr.Textbox(
                        label="Tag to Check",
                        placeholder="e.g. series:my father's wife",
                        interactive=True
                    )
                    cookie_file_tag = gr.File(label="Cookies File (optional)")
                    tag_btn = gr.Button("Check", elem_classes=["process-btn"])
                
                with gr.Column():
                    # Single video result
                    with gr.Row(visible=True) as single_result_row:
                        tag_output = gr.Textbox(
                            label="Result",
                            interactive=False,
                            lines=5,
                            elem_classes=["status-box"]
                        )
                    
                    # Playlist result
                    with gr.Row(visible=False) as playlist_result_row:
                        playlist_tag_status = gr.Textbox(
                            label="Status",
                            value="Ready.",
                            interactive=False,
                            elem_classes=["status-box"]
                        )
                        tag_output_playlist = gr.File(label="Download Results CSV")
            
            # Toggle visibility based on mode
            def toggle_mode(mode):
                if mode == "Single Video":
                    return (
                        gr.update(visible=True),   # single_video_row
                        gr.update(visible=False),  # playlist_row
                        gr.update(visible=True),   # single_result_row
                        gr.update(visible=False)   # playlist_result_row
                    )
                else:
                    return (
                        gr.update(visible=False),  # single_video_row
                        gr.update(visible=True),   # playlist_row
                        gr.update(visible=False),  # single_result_row
                        gr.update(visible=True)    # playlist_result_row
                    )
            
            tag_mode.change(
                fn=toggle_mode,
                inputs=tag_mode,
                outputs=[single_video_row, playlist_row, single_result_row, playlist_result_row]
            )
            
            # Handler function that switches based on mode
            def tag_handler(mode, video_url, playlist_url, tag, cookies):
                if mode == "Single Video":
                    return check_youtube_tag_with_feedback(video_url, tag, cookies), None, None
                else:
                    csv_path, status = check_playlist_tags_with_feedback(playlist_url, tag, cookies)
                    return None, status, csv_path
            
            tag_btn.click(
                fn=tag_handler,
                inputs=[tag_mode, tag_url, playlist_url_tags, tag_input, cookie_file_tag],
                outputs=[tag_output, playlist_tag_status, tag_output_playlist]
            )

        # ── Tab 6: YouTube Filter ────────────────────────────────────────
        with gr.TabItem("🎬 YouTube Filter"):
            gr.Markdown("""
            Filter YouTube videos by duration and type.
            
            **Categories:**
            - **Short Form** — 0 to 7 minutes
            - **Mid Form** — 8 to 15 minutes  
            - **Could be Full Eps** — 15+ minutes
            - **Violators** — Playlists, channels, search results, etc.
            
            Paste video URLs (one per line or comma-separated) and get a categorized CSV report.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    filter_mode = gr.Radio(
                        choices=["Video(s)", "Playlist"],
                        value="Video(s)",
                        label="Input Type",
                        interactive=True
                    )
                    filter_urls_input = gr.Textbox(
                        label="YouTube URLs",
                        placeholder="Paste video or playlist URLs here (one per line or comma-separated):\nhttps://youtube.com/watch?v=...\nhttps://youtube.com/playlist?list=...",
                        lines=8,
                        interactive=True
                    )
                    cookie_file_filter = gr.File(label="Cookies File (optional)")
                    
                    filter_btn = gr.Button("Filter Videos", elem_classes=["process-btn"])
                
                with gr.Column(scale=1):
                    filter_status = gr.Textbox(
                        label="Summary",
                        value="Ready.",
                        interactive=False,
                        lines=8,
                        elem_classes=["status-box"]
                    )
                    
                    filter_csv_output = gr.File(label="Download Report CSV")
            
            def filter_handler(mode, urls, cookies):
                input_mode = "playlist" if mode == "Playlist" else "video"
                csv_path, summary = filter_youtube_urls(urls, input_mode=input_mode, cookies_path=cookies)
                return summary, csv_path
            
            filter_btn.click(
                fn=filter_handler,
                inputs=[filter_mode, filter_urls_input, cookie_file_filter],
                outputs=[filter_status, filter_csv_output]
            )


if __name__ == "__main__":
    interface.launch(share=True)