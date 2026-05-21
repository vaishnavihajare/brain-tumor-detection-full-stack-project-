from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.models import load_model
from PIL import Image

import numpy as np
import io

# Create FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = load_model("brain_model.h5")

# IMPORTANT:
# Keep class order same as train_data.class_indices
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']


# Home API
@app.get("/")
def home():
    return {
        "message": "Brain Tumor Detection API Running"
    }


# Prediction API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Read uploaded image
        contents = await file.read()

        # Open image
        image = Image.open(io.BytesIO(contents))

        # Convert image to RGB
        image = image.convert("RGB")

        # Resize image
        image = image.resize((128, 128))

        # Convert to numpy array
        image = np.array(image)

        # Normalize image
        image = image / 255.0

        # Expand dimensions
        image = np.expand_dims(image, axis=0)

        # Prediction
        prediction = model.predict(image)

        print("Prediction Probabilities:", prediction)

        # Predicted class index
        predicted_index = np.argmax(prediction)

        # Predicted class label
        predicted_class = classes[predicted_index]

        # Confidence score
        confidence = float(np.max(prediction))

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4)
        }

    except Exception as e:

        return {
            "error": str(e)
        }