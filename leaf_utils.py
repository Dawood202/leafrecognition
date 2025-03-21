import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import math

class LeafImageProcessor:

    @staticmethod
    def get_background_color(image):
        image = image.convert("RGB")
        corners = [
            image.getpixel((0, 0)),  # Top-left
            image.getpixel((image.width - 1, 0)),  # Top-right
            image.getpixel((0, image.height - 1)),  # Bottom-left
            image.getpixel((image.width - 1, image.height - 1))  # Bottom-right
        ]
        avg_color = tuple(
            sum(corner[i] for corner in corners) // len(corners) for i in range(3)
        )
        return avg_color
    
    @staticmethod
    def resize_image_for_display(image, max_width=400):
        width, height = image.size
        if width > max_width:
            new_width = max_width
            new_height = int((new_width / width) * height)
            return image.resize((new_width, new_height))
        return image
    
    @staticmethod
    def resize_image_for_display1(image, max_width=400, max_height=400):
        width, height = image.size
        
        # Calculate aspect ratios
        width_ratio = max_width / width
        height_ratio = max_height / height
        
        # Use the smaller ratio to scale the image
        scaling_factor = min(width_ratio, height_ratio, 1)  # Ensure scaling factor is at most 1
        
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        
        return image.resize((new_width, new_height))
    
    @staticmethod
    def rotate_image(image, angle, background_color):
        image=LeafImageProcessor.resize_image_for_display(image)
        img = np.array(image.convert('RGB'))  # Convert PIL image to numpy (RGB)
        

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=background_color)
        return Image.fromarray(rotated)
    
    @staticmethod
    def preprocess_image(image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    @staticmethod
    def extract_leaf_border(image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        leaf_mask = np.ones_like(image) * 255
        cv2.drawContours(leaf_mask, [largest_contour], -1, 0, thickness=10)
        return Image.fromarray(leaf_mask)
    
    @staticmethod
    def split_horizontally(image):
        width, height = image.size
        midpoint = height // 2
        upper_crop = image.crop((0, 0, width, midpoint))
        lower_crop = image.crop((0, midpoint, width, height))
        return LeafImageProcessor.imageBanding_new(upper_crop),LeafImageProcessor.imageBanding_new(lower_crop)
    
    def imageBanding(self,img):
        # if isinstance(img, np.ndarray):
        #     img = Image.fromarray(img)
        width, height = img.size
        new_size = max(width, height)
        dominant_color = self.get_background_color(img)
        new_image = Image.new("RGB", (new_size, new_size), dominant_color)
        new_image.paste(img, ((new_size - width) // 2, (new_size - height) // 2))
        return new_image
    
    @staticmethod
    def imageBanding_new(img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        width, height = img.size
        diagonal = int(math.sqrt(width**2 + height**2))
        if width > height:
            padding = diagonal - width
        else:
            padding = diagonal - height
        new_size = max(width, height) + padding
        dominant_color = LeafImageProcessor.get_background_color(img)
        new_image = Image.new("RGB", (new_size, new_size), dominant_color)
        new_image.paste(img, ((new_size - width) // 2, (new_size - height) // 2))
        return new_image
    
    @staticmethod
    def split_vertically(image):
        width, height = image.size
        midpoint = width // 2
        # return image.crop((0, 0, midpoint, height)), image.crop((midpoint, 0, width, height))
        upper_crop = image.crop((0, 0, midpoint, height))
        lower_crop = image.crop((midpoint, 0, width, height))
        return LeafImageProcessor.imageBanding_new(upper_crop),LeafImageProcessor.imageBanding_new(lower_crop)

    
    @staticmethod
    def split_leaf_border(image):
        apex, base = LeafImageProcessor.split_horizontally(image)
        left_border, right_border = LeafImageProcessor.split_vertically(image)
        return apex, base, left_border, right_border
    
    @staticmethod
    def convert_to_bytes(pil_image):
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    
    @staticmethod
    def get_background_color(image):
        image = image.convert("RGB")
        corners = [
        image.getpixel((0, 0)),  # Top-left
        image.getpixel((image.width - 1, 0)),  # Top-right
        image.getpixel((0, image.height - 1)),  # Bottom-left
        image.getpixel((image.width - 1, image.height - 1))  # Bottom-right
        ]
        avg_color = tuple(
        sum(corner[i] for corner in corners) // len(corners) for i in range(3)
    )
        return avg_color
    
    # def imageBanding(self,img):
    #     if isinstance(img, np.ndarray):
    #         img = Image.fromarray(img)
    #     width, height = img.size
    #     new_size = max(width, height) + 200
    #     dominant_color = self.get_background_color(img)
    #     new_image = Image.new("RGB", (new_size, new_size), dominant_color)
    #     new_image.paste(img, ((new_size - width) // 2, (new_size - height) // 2))
    #     return new_image
    
    @staticmethod
    def preprocess_for_model_old(image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = np.array(image)
        if len(image.shape) == 2:  # If grayscale, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_resized = cv2.resize(image, (224, 224))
            image_normalized = image_resized / 255.0
        return np.expand_dims(image_normalized, axis=0).astype(np.float32)
    
    @staticmethod
    def preprocess_for_model(image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized / 255.0
        return np.expand_dims(image_normalized, axis=0).astype(np.float32)
    
    @staticmethod
    def saveImageFile(uploaded_file,save_path):
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return (f"File saved successfully as {save_path}")
    

    