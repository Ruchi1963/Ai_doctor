import os

import numpy as np
import requests
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

load_dotenv()

MODEL_URL = os.getenv("MODEL_URL")
MODEL_PATH = "models/best_model.h5"

# Download the model only if not already present
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("📥 Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded.")
    else:
        print("✅ Model already exists.")

download_model()
model = load_model(MODEL_PATH, compile=False)

# Class mappings
class_descriptions = {
    'normal': (
        "*Normal:* Cells appear healthy and show no signs of cancer. "
        "There is no indication of abnormal tissue structure or cellular activity. "
        "No further investigation is typically required, but periodic screenings help ensure continued health."
    ),
    'benign': (
        "*Benign:* Cells may appear abnormal but are *non-cancerous* and do not spread to other tissues. "
        "These growths are usually slow-growing and not life-threatening. "
        "Monitoring for changes in size or behavior is common, and removal may be considered if symptoms develop."
    ),
    'malignant': (
        "*Malignant:* Cells are *cancerous*, showing uncontrolled growth, abnormal structure, and potential to invade nearby tissues or spread to distant organs. "
        "Further diagnostic evaluation and staging are important to determine the extent. "
        "Management may involve treatment plans such as surgery, chemotherapy, or radiation depending on the progression."
    )
}

class_labels = {0: 'benign', 1: 'malignant', 2: 'normal'}

def breast_cancer_detection_model(image_path):
    try:
        img = load_img(image_path, target_size=(256, 256))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]

        result = class_descriptions[predicted_class]
        return result

    except Exception as e:
        return f"❌ Error processing the image: {str(e)}"
