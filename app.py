import os
import base64
import time
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

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
model = genai.GenerativeModel('gemini-pro-vision')

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

    filename = secure_filename(image.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    # --- Run CNN (Dummy for Presentation) ---
    print("\n[STEP 1] Running CNN Model (Simulated DenseNet121)...")
    start = time.time()
    cnn_results = cnn_model.predict(filepath)
    print(f"[INFO] CNN Finished in {time.time() - start:.2f} seconds\n")

    # --- Run Gemini API (Real AI for detection + recipes) ---
    print("[STEP 2] Calling Gemini API for Ingredient Detection & Recipe Generation...")

    with open(filepath, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    prompt = f"""
    You are an expert chef and food vision model.
    Given this food image (base64 below), identify key ingredients and suggest 3 recipes.
    Respond in JSON format as:
    {{
      "ai_detected_ingredients": ["ingredient1", "ingredient2", ...],
      "recipes": [
        {{
          "name": "Recipe Name",
          "ingredients": ["ingredient1", "ingredient2", ...],
          "instructions": "Step by step instructions"
        }}
      ]
    }}
    Image (base64): {img_base64[:500]}... (truncated)
    """

    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        response = model.generate_content(prompt)
        ai_response_text = response.text
        print("[INFO] Gemini API response received successfully.")
    except Exception as e:
        print("[ERROR] Gemini API failed:", e)
        return jsonify({"error": "Gemini API failed"}), 500

    return jsonify({
        "cnn_results": cnn_results,
        "ai_response": ai_response_text,
        "image_url": f"/static/uploads/{filename}"
    })


@app.errorhandler(404)
def not_found_error(e):
    return jsonify({"error": "404 Not Found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
