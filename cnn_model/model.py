import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class FoodIngredientCNN:
    def __init__(self, device='cpu'):
        """Initialize the model with preprocessing config"""
        print("[INFO] Initializing DenseNet121 Food Classifier Demo...")
        
        # Set device (CPU/CUDA)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        
        # Demo list of ingredients (subset)
        self.ingredients = [
            "tomato", "onion", "garlic", "chicken", "beef", "pork",
            "carrot", "potato", "rice", "pasta", "olive oil", "butter",
            "salt", "pepper", "basil", "oregano", "cumin", "paprika",
            "mushroom", "bell pepper", "cheese", "egg", "milk", "cream",
            "lemon", "lime", "ginger", "soy sauce", "vinegar", "honey",
            "spinach", "broccoli", "lettuce", "cucumber", "zucchini",
            "eggplant", "cabbage", "celery", "green beans", "asparagus"
        ]
        
        # Standard ImageNet preprocessing configuration
        self.preprocess_config = {
            "resize": (256, 256),
            "center_crop": 224,
            "to_tensor": True,
            "normalize": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
        
        # Build preprocessing pipeline
        self.preprocess = self._build_preprocess_pipeline()
        
        # Initialize model architecture for demonstration
        try:
            self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, 316)  # 316 ingredient classes
            self.model = self.model.to(self.device)
            print("[INFO] Model architecture initialized successfully")
        except Exception as e:
            print(f"[ERROR] Model initialization error: {e}")

    def _build_preprocess_pipeline(self):
        """Build preprocessing pipeline matching training setup"""
        transforms_list = []
        cfg = self.preprocess_config
        
        if cfg["resize"]:
            transforms_list.append(
                transforms.Resize(cfg["resize"], interpolation=transforms.InterpolationMode.BILINEAR)
            )
        if cfg["center_crop"]:
            transforms_list.append(transforms.CenterCrop(cfg["center_crop"]))
        if cfg["to_tensor"]:
            transforms_list.append(transforms.ToTensor())
        if cfg["normalize"]:
            transforms_list.append(transforms.Normalize(mean=cfg["mean"], std=cfg["std"]))
        
        return transforms.Compose(transforms_list)

    def predict(self, image_path):
        """Demo prediction showing model architecture"""
        print("\n=== DenseNet121 Food Ingredient Classifier Demo ===")
        print(f"Model Architecture: DenseNet121")
        print(f"Number of Classes: 316 ingredients")
        print(f"Input Resolution: 224x224 pixels")
        print(f"Device: {self.device}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            preprocessed = self.preprocess(image).unsqueeze(0)  # Add batch dimension
            preprocessed = preprocessed.to(self.device)
            print(f"[INFO] Image preprocessed: {tuple(preprocessed.shape)}")
            
            # For demo purposes, select random ingredients
            num_ingredients = np.random.randint(3, 6)
            detected_ingredients = np.random.choice(
                self.ingredients, 
                size=num_ingredients, 
                replace=False
            )
            confidence_scores = np.random.uniform(0.85, 0.98, size=num_ingredients)
            
            print("\n[INFO] Demo Prediction Complete!")
            print("[NOTE] This is a demonstration of the CNN architecture.")
            print("[NOTE] Actual ingredient detection uses Gemini API.\n")
            
            return {
                "ingredients": detected_ingredients.tolist(),
                "confidence_scores": confidence_scores.tolist()
            }
            
        except Exception as e:
            print(f"[ERROR] Prediction error: {e}")
            return {
                "ingredients": [],
                "confidence_scores": []
            }
