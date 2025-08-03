import gradio as gr
import numpy as np
import cv2
from PIL import Image
from deepface import DeepFace

def detect_emotion(image):
    try:
        # Validate input type
        if image is None:
            return "Error: No image received."

        # Convert PIL to NumPy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Debug output
        if not isinstance(image, np.ndarray):
            return f"Error: Image is not a NumPy array. Got {type(image)}"

        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Handle alpha channel
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        # Convert RGB to BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # DeepFace analysis
        result = DeepFace.analyze(image_bgr, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return f"Detected Emotion: {emotion}"

    except Exception as e:
        return f"Error in image emotion detection: {e}"

iface = gr.Interface(
    fn=detect_emotion,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=gr.Textbox(label="Emotion"),
    title="Facial Emotion Recognizer",
    description="Upload a face image to detect dominant emotion using DeepFace."
)

if __name__ == "__main__":
    iface.launch()
