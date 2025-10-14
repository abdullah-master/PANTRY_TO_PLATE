import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import google.generativeai as genai
from cnn_model.model import FoodIngredientCNN
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    # Initialize the CNN model
    cnn_model = FoodIngredientCNN()
    print("CNN Model initialized successfully!")
except Exception as e:
    print(f"Error initializing CNN model: {e}")
    cnn_model = None

try:
    # Configure Gemini API
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro-vision')
    print("Gemini API configured successfully!")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        # Check if services are available
        if not cnn_model or not model:
            return jsonify({'error': 'Service not fully initialized. Check server logs.'}), 503
            
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Step 1: Run "CNN" prediction (dummy)
        print("\n=== Running CNN Model ===")
        cnn_results = cnn_model.predict(filepath)
        
        # Step 2: Use Gemini API for actual ingredient detection and recipe generation
        print("\n=== Calling Gemini API ===")
        image_data = base64.b64encode(open(filepath, 'rb').read()).decode('utf-8')
        
        prompt = """
        1. First, analyze this image and list all visible ingredients you can identify.
        2. Then, suggest 2-3 possible recipes that could be made using these ingredients.
        
        Format your response as JSON with this structure:
        {
            "detected_ingredients": ["ingredient1", "ingredient2", ...],
            "suggested_recipes": [
                {
                    "name": "Recipe Name",
                    "ingredients": ["ingredient1", "amount1", ...],
                    "instructions": ["step1", "step2", ...]
                },
                ...
            ]
        }
        """
        
        response = model.generate_content([prompt, image_data])
        ai_results = response.text
        
        # Combine results
        return jsonify({
            'cnn_results': cnn_results,
            'ai_results': ai_results,
            'image_path': f'/static/uploads/{filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error. Please try again later.'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

if __name__ == '__main__':
    print("\n=== Initializing Pantry to Plate Application ===")
    print(f"Upload directory: {app.config['UPLOAD_FOLDER']}")
    print("Server Status: Ready")
    app.run(debug=True, port=5000)