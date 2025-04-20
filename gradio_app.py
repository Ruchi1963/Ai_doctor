from dotenv import load_dotenv

load_dotenv()

import os
import shutil

import gradio as gr
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Imports from your custom modules
from brain_of_the_doctor import analyze_image_with_query, encode_image
from breast_cancer_classifer import breast_cancer_detection_model
from voice_of_the_doctor import text_to_speech_with_elevenlabs
from voice_of_the_patient import transcribe_with_groq

# System Prompt
system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
What's in this image?. Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

def contains_hsi_keywords(speech_text):
    keywords = ["hsi", "hyperspectral imaging", "tissue"]
    return any(keyword.lower() in speech_text.lower() for keyword in keywords)

def is_hsi_image(image_path):
    return image_path and image_path.lower().endswith(('.mat', '.npy', '.hdr'))

# Process inputs safely

from PIL import Image


def process_inputs(audio_filepath, image_pil):
    speech_text = ""
    diagnosis = ""
    audio_output_path = "static/final.mp3"

    try:
        if audio_filepath:
            speech_text = transcribe_with_groq(
                GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
                audio_filepath=audio_filepath,
                stt_model="whisper-large-v3"
            )
        else:
            speech_text = "No audio input provided."

        # Save PIL image to a temporary path
        image_filepath = None
        if image_pil:
            image_filepath = "static/temp_image.png"
            image_pil.save(image_filepath)

        if contains_hsi_keywords(speech_text) or (image_filepath and is_hsi_image(image_filepath)):
            print("⚠️ Detected HSI Case - Redirecting to cancer_classifier")
            diagnosis = breast_cancer_detection_model(image_filepath)
        elif image_filepath:
            print("🧠 Using LLM for diagnosis")
            diagnosis = analyze_image_with_query(
                query=system_prompt + " " + speech_text,
                encoded_image=encode_image(image_filepath),
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        else:
            diagnosis = "No image provided for analysis."

        text_to_speech_with_elevenlabs(
            input_text=diagnosis,
            output_filepath=audio_output_path
        )

    except Exception as e:
        print("❌ Error during processing:", e)
        diagnosis = "An error occurred while processing. Please try again."

    return speech_text, diagnosis, audio_output_path if os.path.exists(audio_output_path) else None


def replay_audio():
    audio_path = "static/final.mp3"
    return audio_path if os.path.exists(audio_path) else None

# Use gr.update for clearing UI
def clear_functionality(audio_input, image_input, speech_output, diagnosis_output, audio_output):
    return (
        None,
        None,
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=None)
    )

# UI
with gr.Blocks() as iface:
    gr.Markdown("## 🧑‍⚕️ AI Doctor with Vision, Voice & Cancer Detection")

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Speak your symptoms")
        image_input = gr.Image(type="pil", label="🖼️ Upload medical image")


    with gr.Row():
        speech_output = gr.Textbox(label="📝 Transcribed Text")
        diagnosis_output = gr.Textbox(label="🩺 Doctor's Diagnosis")

    audio_output = gr.Audio(label="🔊 Doctor's Voice")

    with gr.Row():
        analyze_btn = gr.Button("🧠 Analyze & Diagnose")
        replay_btn = gr.Button("🔁 Replay Last Audio")
        clear_btn = gr.Button("🧹 Clear Inputs & Outputs")

    analyze_btn.click(
        fn=process_inputs,
        inputs=[audio_input, image_input],
        outputs=[speech_output, diagnosis_output, audio_output]
    )

    replay_btn.click(
        fn=replay_audio,
        outputs=[audio_output]
    )

    clear_btn.click(
        fn=clear_functionality,
        inputs=[audio_input, image_input, speech_output, diagnosis_output, audio_output],
        outputs=[audio_input, image_input, speech_output, diagnosis_output, audio_output]
    )

iface.launch(debug=True, share=False, show_error=True, prevent_thread_lock=True)

iface.launch(debug=True)
