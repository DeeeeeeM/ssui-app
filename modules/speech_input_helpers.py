import os
import glob
import tempfile
import zipfile

from modules.utils import normalize_file_path
from modules.main import process_media
from modules.video_downloader import download_single_video, download_playlist


def _create_upload_object(file_path):
    class Upload:
        def __init__(self, name):
            self.name = name
    return Upload(file_path)


def _normalize_file_paths(file_input):
    if not file_input:
        return []
    if isinstance(file_input, list):
        return [normalize_file_path(item) for item in file_input if normalize_file_path(item)]
    normalized = normalize_file_path(file_input)
    return [normalized] if normalized else []


def _process_local_single(file_input, model_size, source_lang, model_type,
                           max_chars, max_words, extend_in, extend_out, collapse_gaps,
                           max_lines_per_segment, line_penalty, longest_line_char_penalty,
                           initial_prompt, normalize_audio, use_demucs,
                           enable_translation, target_lang, service, api_key, ollama_host):
    paths = _normalize_file_paths(file_input)
    if not paths:
        return "❌ Error: Please provide a local audio or video file.", None, None, None, None

    upload = _create_upload_object(paths[0])
    audio_out, video_out, transcript, srt_path = process_media(
        model_size, source_lang, upload, model_type,
        max_chars, max_words, extend_in, extend_out, collapse_gaps,
        max_lines_per_segment, line_penalty, longest_line_char_penalty,
        initial_prompt, normalize_audio, use_demucs
    )

    status = f"✅ Processed local file: {os.path.basename(paths[0])}" if transcript else "❌ Error processing local file."
    return status, audio_out, video_out, transcript, srt_path


def _process_local_multiple(file_inputs, model_size, source_lang, model_type,
                             max_chars, max_words, extend_in, extend_out, collapse_gaps,
                             max_lines_per_segment, line_penalty, longest_line_char_penalty,
                             initial_prompt, normalize_audio, use_demucs,
                             enable_translation, target_lang, service, api_key, ollama_host):
    paths = _normalize_file_paths(file_inputs)
    if not paths:
        return "❌ Error: Please provide local files.", None, None, None, None

    transcripts = []
    srt_paths = []
    errors = []
    for path in paths:
        upload = _create_upload_object(path)
        _, _, transcript, srt_path = process_media(
            model_size, source_lang, upload, model_type,
            max_chars, max_words, extend_in, extend_out, collapse_gaps,
            max_lines_per_segment, line_penalty, longest_line_char_penalty,
            initial_prompt, normalize_audio, use_demucs
        )
        if transcript:
            transcripts.append(f"=== {os.path.basename(path)} ===\n{transcript}\n")
        if srt_path:
            srt_paths.append(srt_path)
        if not transcript or not srt_path:
            errors.append(os.path.basename(path))

    if not transcripts:
        return "❌ Error: No files were processed successfully.", None, None, None, None

    combined_transcript = "\n".join(transcripts)
    if len(srt_paths) == 1:
        srt_output_path = srt_paths[0]
    else:
        zip_dir = tempfile.mkdtemp(prefix="ssui_multi_srt_")
        zip_path = os.path.join(zip_dir, "srt_outputs.zip")
        with zipfile.ZipFile(zip_path, "w") as archive:
            for srt_path in srt_paths:
                archive.write(srt_path, arcname=os.path.basename(srt_path))
        srt_output_path = zip_path

    status = f"✅ Processed {len(transcripts)} file(s)."
    if errors:
        status += f" Skipped: {', '.join(errors)}"

    return status, None, None, combined_transcript, srt_output_path


def _process_youtube_single(youtube_url, cookies_path, model_size, source_lang, model_type,
                             max_chars, max_words, extend_in, extend_out, collapse_gaps,
                             max_lines_per_segment, line_penalty, longest_line_char_penalty,
                             initial_prompt, normalize_audio, use_demucs,
                             enable_translation, target_lang, service, api_key, ollama_host):
    if not youtube_url or not youtube_url.strip():
        return "❌ Error: Please provide a YouTube video URL.", None, None, None, None

    cookies_path = normalize_file_path(cookies_path)
    audio_path, download_status = download_single_video(youtube_url, "mp3", "medium", cookies_path)
    if not audio_path:
        return download_status, None, None, None, None

    upload = _create_upload_object(audio_path)
    audio_out, video_out, transcript, srt_path = process_media(
        model_size, source_lang, upload, model_type,
        max_chars, max_words, extend_in, extend_out, collapse_gaps,
        max_lines_per_segment, line_penalty, longest_line_char_penalty,
        initial_prompt, normalize_audio, use_demucs
    )

    status = f"✅ Transcribed YouTube video. {download_status}"
    return status, audio_out, video_out, transcript, srt_path


def _process_youtube_playlist(youtube_playlist_url, cookies_path, model_size, source_lang, model_type,
                              max_chars, max_words, extend_in, extend_out, collapse_gaps,
                              max_lines_per_segment, line_penalty, longest_line_char_penalty,
                              initial_prompt, normalize_audio, use_demucs,
                              enable_translation, target_lang, service, api_key, ollama_host):
    if not youtube_playlist_url or not youtube_playlist_url.strip():
        return "❌ Error: Please provide a YouTube playlist URL.", None, None, None, None

    cookies_path = normalize_file_path(cookies_path)
    playlist_dir, download_status = download_playlist(youtube_playlist_url, "mp3", "medium", cookies_path)
    if not playlist_dir:
        return download_status, None, None, None, None

    audio_files = sorted(glob.glob(os.path.join(playlist_dir, "**", "*.mp3"), recursive=True))
    if not audio_files:
        return "❌ Error: No downloaded audio files found for the playlist.", None, None, None, None

    status, _, _, combined_transcript, srt_output_path = _process_local_multiple(
        audio_files, model_size, source_lang, model_type,
        max_chars, max_words, extend_in, extend_out, collapse_gaps,
        max_lines_per_segment, line_penalty, longest_line_char_penalty,
        initial_prompt, normalize_audio, use_demucs
    )

    return f"✅ Processed YouTube playlist. {download_status} {status}", None, None, combined_transcript, srt_output_path
