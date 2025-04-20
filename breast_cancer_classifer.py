import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load your trained model once globally
model = load_model(r'C:\Users\Ruchir\OneDrive - BENNETT UNIVERSITY\Desktop\ai_doctor_2.0\best_model.h5', compile=False)

# Mapping from index to medical-style interpretation
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

# Internal mapping from model index to class label
class_labels = {0: 'benign', 1: 'malignant', 2: 'normal'}
#image_path= r'C:\Users\Ruchir\OneDrive - BENNETT UNIVERSITY\Desktop\test_images\benign\benign (1).png'
# Main prediction function
def breast_cancer_detection_model(image_path):
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=(256, 256))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        # Format and return interpretation
        result = class_descriptions[predicted_class]
        #result += f"\n📊 *Model Confidence:* {confidence:.2f}%"
        return result

    except Exception as e:
        return f"❌ Error processing the image: {str(e)}"
