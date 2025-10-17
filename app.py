import os
import json
import time
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image

# Import CNN model for demo
from cnn_model.model import FoodIngredientCNN

# Load environment variables and configure
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Flask app setup
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Setup upload directory
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Initialize CNN model for demo
try:
    cnn_model = FoodIngredientCNN()
    print("[INFO] CNN Model Demo initialized successfully!")
except Exception as e:
    print(f"[ERROR] CNN Model initialization failed: {e}")
    cnn_model = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(image.filename):
        return jsonify({"error": "Invalid file format. Please upload a PNG, JPG, or JPEG file."}), 400

    # Get preferences
    diet_preference = request.form.get('diet', 'None')
    allergies_json = request.form.get('allergies', '[]')
    try:
        allergies = json.loads(allergies_json)
    except:
        allergies = []

    filename = secure_filename(image.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    # --- STEP 1: Run CNN Model (Dummy for Presentation) ---
    print("\n" + "="*60)
    print("[STEP 1] Running CNN Model (Simulated DenseNet121)...")
    print("="*60)
    start = time.time()
    cnn_results = cnn_model.predict(filepath) if cnn_model else {"ingredients": [], "confidence_scores": []}
    print(f"[INFO] CNN Processing Complete in {time.time() - start:.2f} seconds")
    print(f"[INFO] CNN Detected: {', '.join(cnn_results['ingredients'])}")
    print("="*60 + "\n")

    # --- STEP 2: Run Gemini API (Real AI for detection + recipes) ---
    print("="*60)
    print("[STEP 2] Calling Gemini Vision API...")
    print("="*60)

    try:
        # Load image for Gemini
        img = Image.open(filepath)
        
        # Initialize correct vision model
        vision_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Build preference context
        preference_text = ""
        if diet_preference and diet_preference != "None":
            preference_text += f"\n- Dietary preference: {diet_preference}"
        if allergies:
            preference_text += f"\n- Allergies to avoid: {', '.join(allergies)}"
        
        # Create detailed prompt
        prompt = f"""You are an expert food vision AI and chef assistant.

Analyze this food image and provide:
1. A comprehensive list of visible ingredients
2. Three creative recipes using these ingredients

{preference_text if preference_text else ""}

IMPORTANT: Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):
{{
  "detected_ingredients": ["ingredient1", "ingredient2", "ingredient3", ...],
  "suggested_recipes": [
    {{
      "name": "Recipe Name",
      "ingredients": ["ingredient1", "ingredient2", ...],
      "instructions": ["Step 1 description", "Step 2 description", ...]
    }}
  ]
}}

Ensure all arrays are properly formatted and the JSON is valid."""

        # Generate content with image
        print("[INFO] Sending request to Gemini Vision API...")
        response = vision_model.generate_content([prompt, img])
        ai_response_text = response.text.strip()
        
        print("[INFO] Gemini API Response Received")
        print(f"[INFO] Response Length: {len(ai_response_text)} characters")
        
        # Clean up response (remove markdown code blocks if present)
        if ai_response_text.startswith("```json"):
            ai_response_text = ai_response_text.replace("```json", "").replace("```", "").strip()
        elif ai_response_text.startswith("```"):
            ai_response_text = ai_response_text.replace("```", "").strip()
        
        # Parse JSON response
        try:
            ai_results = json.loads(ai_response_text)
            print(f"[INFO] Successfully parsed AI response")
            print(f"[INFO] AI Detected: {', '.join(ai_results.get('detected_ingredients', []))}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON Parse Error: {e}")
            print(f"[ERROR] Raw Response: {ai_response_text[:200]}...")
            # Fallback response
            ai_results = {
                "detected_ingredients": ["Unable to parse ingredients"],
                "suggested_recipes": [{
                    "name": "Error: Invalid API Response",
                    "ingredients": ["Please try again"],
                    "instructions": ["The API returned an invalid response format."]
                }]
            }
        
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Gemini API Error: {str(e)}")
        print("="*60 + "\n")
        return jsonify({"error": f"Gemini API failed: {str(e)}"}), 500

    # Return combined results
    return jsonify({
        "cnn_results": cnn_results,
        "ai_results": ai_results,  # Changed from ai_response to ai_results
        "image_url": f"/static/uploads/{filename}"
    })


@app.errorhandler(404)
def not_found_error(e):
    return jsonify({"error": "404 Not Found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🍳 Culinary Vision - Pantry to Plate")
    print("="*60)
    print("[INFO] Starting Flask Application...")
    print("[INFO] Access the app at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True)