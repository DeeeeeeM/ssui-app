import os
import subprocess
import tempfile
import shutil
from pathlib import Path


def normalize_audio(input_path: str, output_path: str) -> str:
    """
    Normalize audio to -14 LUFS using ffmpeg loudnorm (EBU R128).
    Returns the output path on success.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[audio_preprocessing] loudnorm warning: {result.stderr[-500:]}")
        # Fall back to just resampling if loudnorm fails
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            output_path
        ]
        subprocess.run(fallback_cmd, capture_output=True)
    return output_path


def separate_vocals(input_path: str, output_dir: str) -> str | None:
    """
    Use demucs to separate vocals from background music.
    Returns path to the vocals WAV file, or None if demucs is not installed.

    Install demucs with: pip install demucs
    Model used: htdemucs (fast, good quality vocal separation)
    """
    try:
        import demucs.separate  # noqa — just checking it's importable
    except ImportError:
        print("[audio_preprocessing] demucs not installed — skipping vocal separation.")
        print("  Install with: pip install demucs")
        return None

    cmd = [
        "python", "-m", "demucs.separate",
        "--two-stems=vocals",
        "--out", output_dir,
        "--name", "htdemucs",
        input_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[audio_preprocessing] demucs error: {result.stderr[-500:]}")
        return None

    # demucs outputs to: output_dir/htdemucs/<stem_name>/vocals.wav
    stem_name = Path(input_path).stem
    vocals_path = os.path.join(output_dir, "htdemucs", stem_name, "vocals.wav")
    if os.path.exists(vocals_path):
        print(f"[audio_preprocessing] Vocals separated: {vocals_path}")
        return vocals_path

    print(f"[audio_preprocessing] Vocals file not found at expected path: {vocals_path}")
    return None


def preprocess_audio(
    input_path: str,
    use_demucs: bool = False,
    normalize: bool = True,
) -> tuple[str, list[str]]:
    """
    Full preprocessing pipeline:
      1. (Optional) Demucs vocal separation
      2. (Optional) Loudnorm audio normalization to -14 LUFS, 16kHz mono

    Returns:
      - processed_path: path to the final preprocessed audio file
      - temp_files: list of temp paths to clean up after transcription

    Usage in predict.py:
        from modules.audio_preprocessing import preprocess_audio
        processed_path, temp_files = preprocess_audio(str(audio_file), use_demucs=True)
        audio = whisperx.load_audio(processed_path)
        # ... transcribe ...
        for f in temp_files:
            if os.path.exists(f): os.remove(f)
    """
    temp_files = []
    current_path = str(input_path)

    # Step 1: Vocal separation with demucs
    if use_demucs:
        demucs_dir = tempfile.mkdtemp(prefix="ssui_demucs_")
        temp_files.append(demucs_dir)  # cleanup the whole dir later
        vocals_path = separate_vocals(current_path, demucs_dir)
        if vocals_path:
            current_path = vocals_path
        else:
            print("[audio_preprocessing] Skipping demucs — using original audio.")

    # Step 2: Loudnorm normalization + resample to 16kHz mono
    if normalize:
        norm_fd, norm_path = tempfile.mkstemp(suffix="_normalized.wav", prefix="ssui_")
        os.close(norm_fd)
        temp_files.append(norm_path)
        current_path = normalize_audio(current_path, norm_path)

    return current_path, temp_files
