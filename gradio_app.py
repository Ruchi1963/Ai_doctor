import os

import gradio as gr
from dotenv import load_dotenv
from pyngrok import ngrok  # Importing pyngrok to create a public URL

#import subprocess


load_dotenv()

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
    keywords = ["hsi", "hyperspectral", "tissue"]
    return any(keyword.lower() in speech_text.lower() for keyword in keywords)

def is_hsi_image(image_path):
    return image_path and image_path.lower().endswith(('.mat', '.npy', '.hdr'))


def process_inputs(audio_filepath, image_pil):
    speech_text = ""
    diagnosis = ""
    audio_output_path = "static/final.mp3"

    try:
        # Step 1: Transcribe audio if provided
        if audio_filepath:
            speech_text = transcribe_with_groq(
                GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
                audio_filepath=audio_filepath,
                stt_model="whisper-large-v3"
            )

        # Step 2: Prepare image if provided
        image_filepath = None
        encoded_image = None
        if image_pil:
            image_filepath = "static/temp_image.png"
            image_pil.save(image_filepath)
            encoded_image = encode_image(image_filepath)

        # Step 3: HSI detection
        if contains_hsi_keywords(speech_text) or (image_filepath and is_hsi_image(image_filepath)):
            diagnosis = breast_cancer_detection_model(image_filepath)

        # Step 4: Multimodal Diagnosis
        elif encoded_image and speech_text:
            # Case: Image + Audio
            diagnosis = analyze_image_with_query(
                query=system_prompt + " " + speech_text,
                encoded_image=encoded_image,
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        elif encoded_image:
            # Case: Image only
            diagnosis = analyze_image_with_query(
                query=system_prompt,
                encoded_image=encoded_image,
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        elif speech_text:
            # Case: Audio only
            diagnosis = analyze_image_with_query(
                query=system_prompt + " " + speech_text,
                encoded_image=None,  # Important fix
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        else:
            diagnosis = "Please provide at least an image or audio to begin diagnosis."

        # Step 5: Generate voice output
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

def download_audio():
    return "static/final.mp3" if os.path.exists("static/final.mp3") else None

def clear_functionality(audio_input, image_input, speech_output, diagnosis_output, audio_output):
    return (
        None,
        None,
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=None)
    )

# Enhanced UI with spinner
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="indigo"), css=""" 
    body {
        background: linear-gradient(to right, #ccfbf1, #e0f2fe);
    }
    .gr-box {
        border-radius: 20px;
        padding: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        background-color: #ffffff;
    }
    .gr-button {
        font-weight: bold;
        font-size: 16px;
        padding: 12px 20px;
        border-radius: 12px;
        transition: transform 0.1s ease;
    }
    .gr-button:active {
        transform: scale(0.96);
    }
    .gr-textbox textarea {
        font-size: 15px;
        line-height: 1.5;
        background-color: #f8fafc;
    }
    .gr-audio .audio-upload-box {
        display: none !important;
    }
    .loading-spinner {
        font-size: 16px;
        color: #0f766e;
        text-align: center;
        padding: 10px;
    }
    @media (max-width: 768px) {
        .gr-box {
            padding: 8px;
        }
        .gr-button {
            font-size: 14px;
            padding: 10px 14px;
        }
    }
""") as iface:
    gr.Markdown("""<h1 style='text-align: center; color: #0f766e;'>🧑‍⚕ AI Doctor</h1>
<p style='text-align: center; color: #334155;'>An AI tool that listens, sees, and speaks to offer quick medical guidance</p>""")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙 Speak your symptoms", show_label=True)
            image_input = gr.Image(type="pil", label="🖼 Upload Medical Image", show_label=True)

        with gr.Column():
            speech_output = gr.Textbox(label="📝 Transcribed Text", lines=4, interactive=False)
            diagnosis_output = gr.Textbox(label="🩺 Doctor's Diagnosis", lines=4, interactive=False)
            audio_output = gr.Audio(label="🔊 Doctor's Voice", interactive=False)
            download_btn = gr.File(label="⬇ Download Voice Output")

    with gr.Row():
        analyze_btn = gr.Button("🧠 Analyze & Diagnose")
        clear_btn = gr.Button("🧹 Clear")

    # Main button functionality
    analyze_btn.click(
        fn=process_inputs,
        inputs=[audio_input, image_input],
        outputs=[speech_output, diagnosis_output, audio_output]
    )

    # Clear button functionality
    clear_btn.click(
        fn=clear_functionality,
        inputs=[audio_input, image_input, speech_output, diagnosis_output, audio_output],
        outputs=[audio_input, image_input, speech_output, diagnosis_output, audio_output]
    )


if __name__ == "__main__":
    try:
        # Create a public URL using ngrok
        #public_url = ngrok.connect(7865) #while running file make specific port busy then change it to something else like  something else, like 7861 or 8000
        #print("🌐 Public Ngrok URL:", public_url)
        
        # Launch Gradio with the same port
        iface.launch(
            debug=True,
            share=False,  # Keep share=False since ngrok will handle the public URL
            server_port=7865,#just make sure that ngtok port(public_url) and local host port(server port) are same
            server_name="0.0.0.0",
            show_error=True,
            prevent_thread_lock=True,
            pwa=True
        )
    except Exception as e:
        print("❌ Error during Gradio launch:", e)