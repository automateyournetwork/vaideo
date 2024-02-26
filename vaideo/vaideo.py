import os
import io
import cv2
import json
import base64
import tempfile
import requests
import subprocess
import streamlit as st
from io import BytesIO
from openai import OpenAI
from pytube import YouTube
from dotenv import load_dotenv
from PIL import Image as PILImage
from moviepy.editor import VideoFileClip

load_dotenv()

# Load the API keys from an environment variable or secure source
openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_api_key)

st.set_page_config(page_title='Vaideo', page_icon="vaideo.png")

# Add a contact email at the bottom of the page
footer = """
    <div style='text-align: center;'>
        <p>Contact us at <a href='mailto:ptcapo@gmail.com'>ptcapo@gmail.com</a></p>
    </div>
"""

def extract_first_half_frames(video_file):
    return extract_frames(video_file, start_frame=0)

def extract_second_half_frames(video_file):
    # Re-read the video file for the second half
    video_file.seek(0)  # Reset the file pointer to the beginning
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        temp_video.flush()

        video = cv2.VideoCapture(temp_video.name)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        midpoint = total_frames // 2
        video.release()

    return extract_frames(video_file, start_frame=midpoint)

def extract_frames(video_file, start_frame):
    video_file.seek(0)  # Reset the file pointer to the beginning
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        temp_video.flush()

        video = cv2.VideoCapture(temp_video.name)
        if not video.isOpened():
            raise ValueError("Unable to open the video file.")

        fps = video.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            video.release()
            raise ValueError("FPS value is zero, unable to process the video.")

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        if duration > 30:
            video.release()
            raise ValueError("The uploaded video is too long. Please upload a video less than 30 seconds.")

        end_frame = total_frames if start_frame != 0 else total_frames // 2

        script_base64_frames = []
        frame_index = 0
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Set the starting frame

        while video.isOpened():
            success, frame = video.read()
            if not success or frame_index + start_frame >= end_frame:
                break

            if frame_index % 4 == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                base64_frame = base64.b64encode(buffer).decode("utf-8")
                script_base64_frames.append(base64_frame)

            frame_index += 1

        video.release()

    return script_base64_frames

def extract_all_frames(video_file):
    video_file.seek(0)  # Reset the file pointer to the beginning
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        temp_video.flush()

        video = cv2.VideoCapture(temp_video.name)
        if not video.isOpened():
            raise ValueError("Unable to open the video file.")

        fps = video.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            video.release()
            raise ValueError("FPS value is zero, unable to process the video.")

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        if duration > 30:
            video.release()
            raise ValueError("The uploaded video is too long. Please upload a video less than 30 seconds.")

        all_base64_frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            _, buffer = cv2.imencode(".jpg", frame)
            base64_frame = base64.b64encode(buffer).decode("utf-8")
            all_base64_frames.append(base64_frame)

        video.release()

    return all_base64_frames

# Function to generate voice-over script using OpenAI's GPT model
def generate_script(base64_frames, prompt):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                f"These are frames of a video. Create a short voiceover script using the images as inspiration to address the following prompt: { prompt }. Only provide the narration no other introductions or commentary; just the narration. Generate no more than 2000 characters. Instead of rejecting the prompt; try your best if possible.",
                *map(lambda x: {"image": x, "resize": 768}, base64_frames[0::10]),
            ],
        },
    ]

    result = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=prompt_messages,
        max_tokens=1000,
    )
    return result.choices[0].message.content

# Function to generate voice-over script using OpenAI's GPT model
def generate_image_script(base64_frames, prompt):
    prompt_messages = [
        {
            "role": "user",
            "content": [
               f"These are frames of a video. I am not asking you to generate an image. Based on the following prompt please generate the text to describe frames inspired by the following prompt: { prompt }. If you feel the prompt is unsafe please try your best instead of rejecting it.",
                *map(lambda x: {"image": x, "resize": 768}, base64_frames[0::10]),
            ],
        },
    ]

    result = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=prompt_messages,
        max_tokens=500,
    )
    return result.choices[0].message.content

def reword_prompt_for_safety(original_prompt):
    # Implement logic to reword the prompt
    # For now, just notifying the script generation of the issue
    revised_prompt = f"This prompt '{original_prompt}' was rejected for safety reasons. Can you reword it but keep the original themes and ideas, just make it safe?"
    return revised_prompt

# Define available voices
voice_options = {
    "Alloy - Male": "alloy",
    "Echo - Male": "echo",
    "Fable - Male (European)": "fable",
    "Onyx - Male (Deeper voice)": "onyx",
    "Nova - Female": "nova",
    "Shimmer - Female": "shimmer"
}

# Function to synthesize speech from the script with a selected voice
def synthesize_speech(script, voice):
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {openai_api_key}"},
        json={"model": "tts-1-hd", "input": script, "voice": voice},
    )
    if response.ok:
        return response.content
    else:
        raise Exception("Error with the audio generation request: " + response.text)

def get_audio_duration(audio_file_path):
    # Command to get the duration of the audio file
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 
        'format=duration', '-of', 
        'default=noprint_wrappers=1:nokey=1', 
        audio_file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def create_video_clip_ffmpeg(frames_dir, audio_file_path, output_video_path):
    # Get the duration of the audio file
    audio_duration = get_audio_duration(audio_file_path)

    # Calculate the number of times the video needs to be looped
    # Assuming each frame represents 1/30th of a second
    num_frames = len([name for name in os.listdir(frames_dir) if name.endswith(".png")])
    video_duration = num_frames / 24
    loop_times = int(audio_duration / video_duration) + 1

    # Command to create a looped video from frames
    cmd_video = [
        'ffmpeg', '-stream_loop', str(loop_times - 1),
        '-framerate', '24', '-i', 
        os.path.join(frames_dir, 'frame_%05d.png'),
        '-threads', '2',
        '-c:v', 'libx264', '-profile:v', 'high', 
        '-crf', '20', '-pix_fmt', 'yuv420p', 
        '-t', str(audio_duration), '-y', 
        os.path.join(frames_dir, 'temp_video.mp4')
    ]
    subprocess.run(cmd_video, check=True)

    # Command to add audio to the video
    cmd_audio = [
        'ffmpeg', '-i', os.path.join(frames_dir, 'temp_video.mp4'),
        '-i', audio_file_path, '-threads', '2', 
        '-c:v', 'copy', '-c:a', 'aac',
        '-strict', 'experimental', '-y', output_video_path
    ]

    subprocess.run(cmd_audio, check=True)

    # Clean up temporary files
    os.remove(os.path.join(frames_dir, 'temp_video.mp4'))
    os.remove(audio_file_path)

def save_frames_as_images(frames, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, frame in enumerate(frames):
        image_data = base64.b64decode(frame)
        image = PILImage.open(io.BytesIO(image_data))
        image.save(os.path.join(output_dir, f"frame_{i:05d}.png"))

def generate_image(prompt, retry_count=3):
    while retry_count > 0:
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            return image_url
        except Exception as e:
            # Check if the exception has a 'response' attribute with a 400 status code
            if hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 400:
                reworded_prompt = reword_prompt_for_safety(prompt)
                prompt = reworded_prompt
                retry_count -= 1
            else:
                raise  # Re-raise the exception if it's not the specific case we're handling

    raise ValueError("Prompt could not be processed after multiple attempts due to safety concerns. Please try adjusting your prompt.")

# Function to download and clip a portion of a YouTube video
def download_youtube_video(youtube_url, start_time, end_time):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension='mp4').first()
    temp_video_path = stream.download()
    clip = VideoFileClip(temp_video_path).subclip(start_time, end_time)
    clip_path = "temp_clip.mp4"
    clip.write_videofile(clip_path)
    os.remove(temp_video_path)  # Remove the original download
    return clip_path

def home_page():
    st.title("Welcome to Vaideo!")
    st.image("vaideo.png")
    st.write("This app provides custom AI-generated voice overs and cartoons for your videos.")

    def go_to_voice_over():
        st.session_state.current_page = "voice_over"

    def go_to_create_cartoon():
        st.session_state.current_page = "create_cartoon"

    # Use the 'on_click' parameter to specify the callback function
    st.button("AI Voice Over", on_click=go_to_voice_over)
    st.button("Create AI Image", on_click=go_to_create_cartoon)

    st.markdown(footer, unsafe_allow_html=True)

# Function for the voice over page
def voice_over_page():
    st.title('Vaideo: Generate Voice AI Voice Overs')
    st.image("vaideo.png")

    # Navigation buttons
    st.button("Create Cartoon", on_click=go_to_create_cartoon)

    video_source = st.radio("Choose your video source", ('Upload', 'YouTube'))
    
    youtube_video_path = None
    uploaded_file = None

    if video_source == 'Upload':
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov"], key="file_uploader")
    elif video_source == 'YouTube':
        youtube_url = st.text_input("Enter YouTube URL")
        start_time = st.number_input("Start Time (seconds)", min_value=0, step=1)
        duration = st.slider("Duration (seconds)", min_value=5, max_value=29, value=29)
   
    # Dropdown for voice selection
    selected_voice = st.selectbox("Select a voice", list(voice_options.keys()), index=0)

    # Read and display conversation starters
    with open('voice_over_starters.json', 'r') as file:
        voice_over_starters = json.load(file)

    display_names = [f"{starter['category']} - {starter['display']}" for starter in voice_over_starters]
    display_names.insert(0, "")
    selected_display = st.selectbox("Choose a conversation starter (optional)", display_names)
    selected_display_name = selected_display.split(" - ", 1)[1] if " - " in selected_display else ""
    selected_prompt = ""
    for starter in voice_over_starters:
        if starter["display"] == selected_display_name:
            selected_prompt = starter["prompt"]
            break

    prompt = st.text_area("Enter your voice-over script prompt", value=selected_prompt)

    if st.button("Generate Voice-over"):
        if video_source == 'YouTube' and youtube_url:
            with st.spinner('Extracting frames from YouTube...'):
                end_time = start_time + duration
                youtube_video_path = download_youtube_video(youtube_url, start_time, end_time)
                uploaded_file = open(youtube_video_path, 'rb')

        if uploaded_file is not None and prompt:
            # Overall process spinner
            with st.spinner('Starting the voice-over generation process (this overall process will take a few minutes depending on the size of your original video)...'):
                try:
                    # Frame extraction spinner
                    with st.spinner('Extracting frames from the video...'):
                        # Extract frames from the first half
                        first_half_frames = extract_first_half_frames(uploaded_file)
                        # Extract frames from the second half
                        second_half_frames = extract_second_half_frames(uploaded_file)

                    # Script generation spinner
                    with st.spinner('Generating the script part one...'):
                        first_half_script = generate_script(first_half_frames, prompt)

                    with st.spinner('Generating the script part two...'):
                        second_half_script = generate_script(second_half_frames, prompt)

                    # Combine frames from both halves for video processing
                    all_base64_frames = extract_all_frames(uploaded_file)

                    # Combine the scripts
                    script = first_half_script + "\n" + second_half_script

                    # Speech synthesis spinner
                    with st.spinner('Synthesizing speech...'):
                        voice_id = voice_options[selected_voice]
                        speech_audio = synthesize_speech(script, voice_id)

                    # Video processing spinner
                    with st.spinner('Processing video (this will take a few minutes)...'):
                        with tempfile.TemporaryDirectory() as temp_dir:
                            save_frames_as_images(all_base64_frames, temp_dir)

                            # Write the audio to a temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_audio:
                                tmp_audio.write(speech_audio)
                                tmp_audio_path = tmp_audio.name

                            # Create the video clip using ffmpeg
                            video_path = os.path.join(temp_dir, 'output.mp4')
                            create_video_clip_ffmpeg(temp_dir, tmp_audio_path, video_path)

                            st.video(video_path)
                            st.markdown(f"{ script }")
                            st.session_state['paid'] = False

                            with open(video_path, "rb") as file:
                                if st.download_button("Download Video", file, file_name="vaideo.mp4"):
                                    st.session_state['paid'] = False
                                    if os.path.exists(video_path):
                                        os.remove(video_path)
                                else:
                                    # Cleanup in case the video is not downloaded
                                    if os.path.exists(video_path):
                                        os.remove(video_path)                                    

                            st.markdown(f"- [Thank you! Please return to Vaideo.ca](https://www.vaideo.ca)", unsafe_allow_html=True)
                                
                except ValueError as e:
                    st.error(str(e))  # Display the error message
                    if os.path.exists(video_path):
                        os.remove(video_path)

            # Close and delete the temporary YouTube video file
            if video_source == 'YouTube' and uploaded_file:
                uploaded_file.close()
                if youtube_video_path and os.path.exists(youtube_video_path):
                    os.remove(youtube_video_path)

    st.markdown(footer, unsafe_allow_html=True)

# Function for the create cartoon page
def create_cartoon_page():
    st.title("Vaideo: Create AI Generated Images Inspired by Video")
    st.image("vaideo.png")
    # Navigation buttons
    st.button("AI Voice Over", on_click=go_to_voice_over)

    video_source = st.radio("Choose your video source", ('Upload', 'YouTube'))
    youtube_video_path = None  # Variable to track the YouTube video file path

    uploaded_file = None
    if video_source == 'Upload':
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])
    else:
        youtube_url = st.text_input("Enter YouTube URL")
        start_time = st.number_input("Start Time (seconds)", min_value=0, step=1)
        duration = st.slider("Duration (seconds)", min_value=5, max_value=29, value=29)

    # Read the JSON file for artistic starters
    with open('artistic_starters.json', 'r') as file:
        artistic_starters = json.load(file)

    display_names = [starter["display"] for starter in artistic_starters]
    display_names.insert(0, "")
    selected_display = st.selectbox("Choose an artistic style (optional)", display_names)

    # Finding the corresponding prompt
    selected_prompt = ""
    for starter in artistic_starters:
        if starter["display"] == selected_display:
            selected_prompt = starter["prompt"]
            break

    user_prompt = st.text_area("Enter a base prompt for the image", value=selected_prompt)

    if st.button("Generate AI Inspired Image"):
        if video_source == 'YouTube' and youtube_url:
            with st.spinner('Extracting frames from YouTube...'):
                end_time = start_time + duration
                youtube_video_path = download_youtube_video(youtube_url, start_time, end_time)
                uploaded_file = open(youtube_video_path, 'rb')

        if uploaded_file is not None and user_prompt:
            # Overall process spinner
            with st.spinner('Starting the cartoon generation process...'):
                try:
                    # Frame extraction spinner
                    with st.spinner('Extracting frames from the video...'):
                        # Extract frames from the first half
                        first_half_frames = extract_first_half_frames(uploaded_file)
                        # Extract frames from the second half
                        second_half_frames = extract_second_half_frames(uploaded_file)

                    # AI prompt generation spinner for the first half
                    with st.spinner('Generating AI text prompt for the first half...'):
                        first_half_ai_prompt = generate_image_script(first_half_frames, user_prompt)

                    # AI prompt generation spinner for the second half
                    with st.spinner('Generating AI text prompt for the second half...'):
                        second_half_ai_prompt = generate_image_script(second_half_frames, user_prompt)

                    # Combine the AI prompts
                    combined_ai_prompt = first_half_ai_prompt + "\n" + second_half_ai_prompt

                    # Cartoon image generation spinner
                    with st.spinner('Generating cartoon image...'):
                        try:
                            # Use the combined_ai_prompt to generate the cartoon image
                            image_url = generate_image(combined_ai_prompt)
                        except ValueError as e:
                            st.error(str(e))  # If the prompt fails after retries, show an error
                            return  # Stop further execution in case of failure

                    # Displaying the result spinner
                    with st.spinner('Finalizing and displaying the image...'):
                        response = requests.get(image_url)
                        image = PILImage.open(BytesIO(response.content))
                        buf = BytesIO()
                        image.save(buf, format="PNG")
                        byte_im = buf.getvalue()                        
                        st.image(image_url, caption=combined_ai_prompt)
                        st.session_state['paid'] = False

                    # Download button
                    if st.download_button(label="Download Image",
                        data=byte_im,
                        file_name="vaideo.png",
                        mime="image/png"):
                        st.session_state['paid'] = False  # Reset payment status

                    st.markdown(f"- [Thank you! Please return to Vaideo.ca](https://www.vaideo.ca)", unsafe_allow_html=True)

                except ValueError as e:
                    st.error(str(e))  # Display the error message
    
        # Close and delete the temporary YouTube video file
        if video_source == 'YouTube' and uploaded_file:
            uploaded_file.close()
            if youtube_video_path and os.path.exists(youtube_video_path):
                os.remove(youtube_video_path)                            
    
    st.markdown(footer, unsafe_allow_html=True)

def go_to_home():
    st.session_state.current_page = "home"

def go_to_voice_over():
    st.session_state.current_page = "voice_over"

def go_to_create_cartoon():
    st.session_state.current_page = "create_cartoon"

# Updated main function
def main():
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"

    # Render pages based on current state
    if st.session_state.current_page == "home":
        home_page()
    elif st.session_state.current_page == "voice_over":
        voice_over_page()
    elif st.session_state.current_page == "create_cartoon":
        create_cartoon_page()

if __name__ == "__main__":
    main()