"""
SRT translation module supporting multiple LLM and translation service choices.
Supports: Google Translate, DeepL, OpenAI GPT, Deepseek, and local LLMs.
"""

import os
import re
import tempfile
from typing import Optional, Tuple
import requests


def parse_srt(srt_content: str) -> list:
    """Parse SRT file content into subtitle blocks."""
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:.*\n)*?)(?=\n\d+\n|\Z)'
    matches = re.findall(pattern, srt_content)
    
    subtitles = []
    for index, start, end, text in matches:
        subtitles.append({
            'index': index,
            'start': start,
            'end': end,
            'text': text.strip()
        })
    return subtitles


def write_srt(subtitles: list) -> str:
    """Convert subtitle blocks back to SRT format."""
    srt_lines = []
    for sub in subtitles:
        srt_lines.append(sub['index'])
        srt_lines.append(f"{sub['start']} --> {sub['end']}")
        srt_lines.append(sub['text'])
        srt_lines.append('')
    return '\n'.join(srt_lines)


def translate_google(text: str, target_lang: str) -> str:
    """Translate using Google Translate (free)."""
    try:
        from google_trans_new import google_translator
        translator = google_translator()
        return translator.translate(text, lang_tgt=target_lang)
    except ImportError:
        return f"[Error: google-trans-new not installed. Install via: pip install google-trans-new]\n{text}"
    except Exception as e:
        return f"[Google Translate Error: {str(e)}]\n{text}"


def translate_deepl(text: str, target_lang: str, api_key: Optional[str] = None) -> str:
    """Translate using DeepL API (free tier available)."""
    try:
        import deepl
        auth_key = api_key or os.environ.get('DEEPL_API_KEY')
        if not auth_key:
            return f"[Error: DeepL API key not provided or DEEPL_API_KEY env var not set]\n{text}"
        
        translator = deepl.Translator(auth_key)
        # Map language codes
        lang_map = {
            'en': 'EN-US', 'es': 'ES', 'fr': 'FR', 'de': 'DE',
            'it': 'IT', 'pt': 'PT-BR', 'ru': 'RU', 'ja': 'JA',
            'ko': 'KO', 'zh': 'ZH', 'ar': 'AR', 'tr': 'TR'
        }
        target = lang_map.get(target_lang, target_lang.upper())
        result = translator.translate_text(text, target_lang=target)
        return str(result)
    except ImportError:
        return f"[Error: deepl not installed. Install via: pip install deepl]\n{text}"
    except Exception as e:
        return f"[DeepL Error: {str(e)}]\n{text}"


def translate_openai(text: str, target_lang: str, api_key: Optional[str] = None) -> str:
    """Translate using OpenAI GPT (requires API key)."""
    try:
        import openai
        api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return f"[Error: OpenAI API key not provided or OPENAI_API_KEY env var not set]\n{text}"
        
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Translate to {target_lang}. Return only the translated text."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
        )
        return response['choices'][0]['message']['content']
    except ImportError:
        return f"[Error: openai not installed. Install via: pip install openai]\n{text}"
    except Exception as e:
        return f"[OpenAI Error: {str(e)}]\n{text}"


def translate_deepseek(text: str, target_lang: str, api_key: Optional[str] = None) -> str:
    """Translate using Deepseek API (free tier available)."""
    try:
        api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            return f"[Error: Deepseek API key not provided or DEEPSEEK_API_KEY env var not set]\n{text}"
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'system', 'content': f'Translate to {target_lang}. Return only the translated text.'},
                {'role': 'user', 'content': text}
            ],
            'temperature': 0.3
        }
        response = requests.post(
            'https://api.deepseek.com/chat/completions',
            json=payload,
            headers=headers,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"[Deepseek API Error {response.status_code}: {response.text}]\n{text}"
    except Exception as e:
        return f"[Deepseek Error: {str(e)}]\n{text}"


def translate_local_llm(text: str, target_lang: str, ollama_host: str = "http://localhost:11434") -> str:
    """Translate using local LLM via Ollama (requires Ollama running locally)."""
    try:
        payload = {
            'model': 'llama2',  # or other available model
            'prompt': f'Translate to {target_lang}. Return only the translated text:\n{text}',
            'stream': False
        }
        response = requests.post(
            f'{ollama_host}/api/generate',
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get('response', text)
        else:
            return f"[Local LLM Error {response.status_code}]\n{text}"
    except Exception as e:
        return f"[Local LLM Error: {str(e)}]\n{text}"


def translate_srt(
    srt_file_path: str,
    target_lang: str,
    service: str,
    api_key: Optional[str] = None,
    ollama_host: str = "http://localhost:11434"
) -> Tuple[Optional[str], str]:
    """
    Translate an SRT file using the specified service.
    
    Args:
        srt_file_path: Path to the SRT file
        target_lang: Target language code (e.g., 'es', 'fr', 'de')
        service: Translation service ('google', 'deepl', 'openai', 'deepseek', 'local_llm')
        api_key: API key for paid services
        ollama_host: URL for local LLM (Ollama)
    
    Returns:
        Tuple of (output_file_path, status_message)
    """
    try:
        # Read SRT file
        with open(srt_file_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        
        # Parse subtitles
        subtitles = parse_srt(srt_content)
        
        if not subtitles:
            return None, "Error: Could not parse any subtitles from the SRT file."
        
        # Translate based on service
        translate_fn = {
            'google': translate_google,
            'deepl': translate_deepl,
            'openai': translate_openai,
            'deepseek': translate_deepseek,
            'local_llm': translate_local_llm
        }.get(service.lower())
        
        if not translate_fn:
            return None, f"Error: Unknown translation service '{service}'"
        
        # Translate each subtitle
        translated_count = 0
        error_count = 0
        
        for sub in subtitles:
            original_text = sub['text']
            if service == 'local_llm':
                translated = translate_local_llm(original_text, target_lang, ollama_host)
            elif service in ['deepl', 'openai', 'deepseek']:
                translated = translate_fn(original_text, target_lang, api_key)
            else:
                translated = translate_fn(original_text, target_lang)
            
            # Check if translation failed (has error marker)
            if '[Error:' in translated or '[' in translated and 'Error' in translated:
                error_count += 1
                sub['text'] = original_text  # Keep original on error
            else:
                sub['text'] = translated
                translated_count += 1
        
        # Write translated SRT
        translated_srt = write_srt(subtitles)
        
        # Save to temp file
        fd, output_path = tempfile.mkstemp(suffix=".srt", text=True)
        os.close(fd)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_srt)
        
        status = f"✅ Translation complete: {translated_count}/{len(subtitles)} subtitles translated"
        if error_count > 0:
            status += f" ({error_count} errors, originals kept)"
        
        return output_path, status
        
    except Exception as e:
        return None, f"Error translating SRT: {str(e)}"
