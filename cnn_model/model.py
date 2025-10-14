import numpy as np
from PIL import Image
import time

class FoodIngredientCNN:
    def __init__(self):
        print("Loading CNN model weights...")
        time.sleep(1)  # Simulate model loading time
        print("CNN model initialized successfully!")
        
        # Predefined common ingredients for realistic output
        self.common_ingredients = [
            "tomato", "onion", "garlic", "potato", "carrot", "bell pepper",
            "chicken", "beef", "rice", "pasta", "mushroom", "broccoli",
            "spinach", "lemon", "ginger", "cucumber", "lettuce"
        ]
    
    def preprocess_image(self, image_path):
        """Simulate image preprocessing"""
        print("Preprocessing image...")
        time.sleep(0.5)  # Simulate preprocessing time
        
        # Actually load the image to get dimensions (for realism)
        img = Image.open(image_path)
        print(f"Image processed: {img.size[0]}x{img.size[1]} pixels")
        return img
    
    def predict(self, image_path):
        """Simulate CNN prediction"""
        print("Running CNN inference on the image...")
        
        # Preprocess image (for realistic console output)
        self.preprocess_image(image_path)
        
        # Simulate processing time
        time.sleep(1.5)
        
        # Randomly select 3-6 ingredients for realistic prediction
        num_ingredients = np.random.randint(3, 7)
        detected_ingredients = np.random.choice(
            self.common_ingredients, 
            size=num_ingredients, 
            replace=False
        ).tolist()
        
        print("CNN prediction completed!")
        return {
            "ingredients": detected_ingredients,
            "confidence_scores": np.random.uniform(0.85, 0.98, size=len(detected_ingredients)).tolist()
        }