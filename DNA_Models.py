import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
import streamlit as st


@st.cache_resource
def load_trained_models_once(model_paths):
    """Load all models once and store them in a dictionary."""
    models = {}
    for path in model_paths:
        try:
            models[path] = load_model(path)
        except Exception as e:
            st.error(f"Error loading model from path {path}: {e}")
    return models


class DNA_Models:
    def __init__(self, model_paths):
        """Initialize the ModelHandler with model paths and load models."""
        self.model_paths = model_paths
        self.models = load_trained_models_once(list(model_paths.keys()))

    def predict_and_find_highest_confidence(self, model_name, image):
        """
        Predict the class and confidence for an image using the specified model
        and find the highest confidence image from the dataset.
        """
        model_map = {
            "Apex": r"C:\DNA\codes\ModelCODE\Drawingscanned_apex_20241210.keras",
            "Base": r"C:\DNA\codes\ModelCODE\Drawingscanned_base_20241210.keras",
            "Margin": r"C:\DNA\codes\ModelCODE\Drawingscanned_margin_20241210.keras",
            "Shape": r"C:\DNA\codes\ModelCODE\Drawingscanned_shape_20241218.keras"
        }

        if model_name not in model_map:
            raise ValueError(f"Model name '{model_name}' is not recognized.")
        
        model_path = model_map[model_name]
        model = self.models.get(model_path)

        if not model:
            raise ValueError(f"Model for '{model_name}' is not loaded.")

        #print(f"Prepared input shape: {image.shape}, dtype: {image.dtype}")

        # Predict the class
        try:
            dataset_path = self.model_paths.get(model_path)
            categories = np.sort(os.listdir(dataset_path))
            prediction = model.predict(image)
            confidence = prediction[0, np.argmax(prediction)]
            predicted_class=categories[np.argmax(prediction)]
            print(predicted_class)
            
        except Exception as e:
            print(e)
            st.error(f"Error during prediction: {e}")
            return None, None, None, None

        # Find the matching image from the dataset
        
        if not dataset_path:
            st.error(f"Dataset path not found for model: {model_name}")
            return predicted_class, confidence, None, None

        class_folder = os.path.join(dataset_path, str(predicted_class))
        if not os.path.exists(class_folder):
            st.warning(f"Class folder not found: {class_folder}")
            return predicted_class, confidence, None, None

        # Process images in the class folder
        images = [img for img in os.listdir(class_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            st.warning(f"No valid images found in {class_folder}")
            return predicted_class, confidence, None, None

        SIZE = 224
        confidence_image_path = None
        confidence_image_id = None
        for img_name in images[:100]:  # Limit to 100 images for faster processing
            image_path = os.path.join(class_folder, img_name)
            img = cv2.imread(image_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (SIZE, SIZE))
            img_array = np.array([img_resized / 255.0])

            try:
                image_prediction = model.predict(img_array)
                image_confidence = image_prediction[0, np.argmax(image_prediction)]
                if image_confidence >= confidence:
                    confidence_image_path = image_path
                    confidence_image_id = os.path.splitext(img_name)[0]
                    break
            except Exception as e:
                st.error(f"Error during dataset image prediction: {e}")
                continue

        return predicted_class, confidence, confidence_image_id, confidence_image_path


# def preprocess_image(image):
#     """Preprocess the image for leaf border extraction."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     kernel = np.ones((5, 5), np.uint8)
#     morph_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     return morph_image




# Define model paths
# model_paths = {
#     r"E:\DNA_models\Drawingscanned_apex_20241210.keras": r"E:\DNA_models\drawing_apex",
#     r"E:\DNA_models\Drawingscanned_base_20241210.keras": r"E:\DNA_models\drawing_base",
#     r"E:\DNA_models\Drawingscanned_margin_20241210.keras": r"E:\DNA_models\drawing_margin",
#     r"E:\DNA_models\Drawingscanned_shape_harita_v2_latest.keras": r"E:\DNA_models\drawing_shape"
# }

# Initialize the handler
    #handler = DNA_Models(model_paths)

# Process the input image
    #input_image_path = r"D:\Projects\Python\DNAUtilities\RotationStraightOutput\Apex.jpg"
    #img = cv2.imread(input_image_path)


#preprocessed_image = preprocess_image(img)
    #leaf_border_image = extract_leaf_border(preprocessed_image)
#prepared_image = preprocess_for_model(preprocessed_image)


# Predict and get results

