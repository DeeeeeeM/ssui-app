from modules.languages import WHISPER_LANGUAGES
from modules.playlist_extractor import extract_playlist_to_csv 
from modules.tag_checker import check_playlist_tags, check_youtube_tag
from modules.sub_dl import download_srt
from modules.main import process_media

import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gradio as gr 

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

    UPDATE: The app now includes Youtube metadata extraction features: (title / URL / ID, subtitles, tag checking)

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
                            value="large-v2",
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
            cookie_file_extract = gr.File(label="YouTube Cookies File (optional)", file_types=None, interactive=True)
            process_btn = gr.Button("Process")
            csv_output = gr.File(label="Download CSV")
            process_btn.click(
                extract_playlist_to_csv,
                inputs=[playlist_url, cookie_file_extract],
                outputs=csv_output
            )

        with gr.TabItem("SRT Downloader"):
            gr.Markdown("### Download English subtitles (.srt) from a YouTube video(s). <i>Separate each URL with a comma or Enter for multiple videos.</i>")

            srt_url = gr.Textbox(label="YouTube Video URL", placeholder="Paste video URL here")
            cookie_file_srt = gr.File(label="YouTube Cookies File (optional)", file_types=None, interactive=True)
            srt_btn = gr.Button("Process")
            srt_file = gr.File(label="Download SRT")
            srt_status = gr.Textbox(label="Status", interactive=False)
            srt_btn.click(
                download_srt,
                inputs=[srt_url, cookie_file_srt],
                outputs=[srt_file, srt_status]
            )

        with gr.TabItem("Tag Checker"):
            gr.Markdown("### Check if a specific tag exists in a YouTube video's metadata.")
            gr.Markdown("*Tip: If a video is age-restricted or otherwise requires authentication, export cookies from your browser (cookies.txt) and upload it below.*")
            gr.Markdown("*How to export cookies: Install the 'Get cookies.txt' extension in your browser, sign into YouTube in the browser, then export using the extension and upload the cookies file here.*")
            tag_url = gr.Textbox(label="YouTube Video URL", placeholder="Paste video URL here")
            tag_input = gr.Textbox(label="Tag to Check", placeholder="Type the tag (e.g. series:my father's wife)")
            cookie_file_tag = gr.File(label="YouTube Cookies File (optional)", file_types=None, interactive=True)
            tag_btn = gr.Button("Process")
            tag_output = gr.Textbox(label="Tag Check Result", interactive=False)
            tag_btn.click(
                check_youtube_tag,
                inputs=[tag_url, tag_input, cookie_file_tag],
                outputs=tag_output
            )

        with gr.TabItem("Playlist Tag Checker"):

            gr.Markdown(
                """ 
                Check if a specific tag exists in all videos of a YouTube playlist.

                <b><i>Note: The process may take longer due to the number of videos being checked.</i></b>
                """
            )
            gr.Markdown("*Tip: If some videos are age-restricted, upload a cookies.txt file so the app can check them.*")
            gr.Markdown("*How to export cookies: Install the 'Get cookies.txt' extension in your browser, sign into YouTube in the browser, then export using the extension and upload the cookies file here.*")
            playlist_url_tags = gr.Textbox(label="YouTube Playlist URL", placeholder="Paste playlist URL here")
            tag_input_playlist = gr.Textbox(label="Tag to Check", placeholder="Type the tag (e.g. series:my father's wife)")
            cookie_file_playlist = gr.File(label="YouTube Cookies File (optional)", file_types=None, interactive=True)
            tag_btn_playlist = gr.Button("Process")
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
        let outputs = document.querySelectorAll("textarea, input[type='file'], video, audio");
        outputs.forEach(function(output) {
            output.addEventListener("change", playNotify);
        });
    });
    </script>
    """
)
    
if __name__ == "__main__":
    interface.launch(share=True)