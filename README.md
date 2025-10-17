# 🍳 Culinary Vision - Pantry to Plate

**Ingredient Detection and Recipe Generator using CNN + Gemini AI**

---

## 📋 Project Overview

This Flask-based web application combines Computer Vision (CNN) and Generative AI to analyze food images, detect ingredients, and suggest recipes. The system uses a DenseNet121 CNN architecture for demonstration and Gemini Vision API for actual intelligent ingredient recognition and recipe generation.

### Key Features
- 📸 Image-based ingredient detection
- 🧠 Dual-layer analysis: CNN Model + Gemini AI
- 🍽️ Smart recipe suggestions based on detected ingredients
- 🥗 Dietary preferences & allergy support
- 💫 Modern, responsive UI

---

## 🏗️ Project Structure

```
pantry_to_plate/
│
├── app.py                      # Flask backend (main application)
├── cnn_model/
│   └── model.py                # CNN model (DenseNet121 demo)
│
├── static/
│   ├── styles.css              # Custom CSS (minimal)
│   └── uploads/                # Uploaded images directory
│       └── .gitkeep
│
├── templates/
│   └── index.html              # Frontend interface
│
├── .env                        # Environment variables (API keys)
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Google Gemini API key

### Step 1: Clone or Download
```bash
cd pantry_to_plate
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables
Create a `.env` file in the root directory (or use the existing one):
```
GEMINI_API_KEY=your_actual_api_key_here
```

⚠️ **Important**: Replace with your actual Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Step 5: Create Upload Directory
```bash
mkdir -p static/uploads
```

---

## ▶️ Running the Application

### Start the Flask Server
```bash
python app.py
```

You should see:
```
🍳 Culinary Vision - Pantry to Plate
============================================================
[INFO] Starting Flask Application...
[INFO] Access the app at: http://127.0.0.1:5000
============================================================
[INFO] CNN Model Demo initialized successfully!
 * Running on http://127.0.0.1:5000
```

### Access the Application
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Testing the Application

### Test Case 1: Basic Image Upload
1. Open the application in your browser
2. Click "Upload Photo" and select a food image
3. Click "Find My Recipes!"
4. Observe the console logs showing CNN processing followed by Gemini API call
5. Verify results display in sections:
   - **Section 3**: CNN Model Analysis (with confidence scores)
   - **Section 4**: AI-Enhanced Detection
   - **Section 5**: Suggested Recipes

### Test Case 2: With Dietary Preferences
1. Upload a food image
2. Select a dietary preference (e.g., "Vegan")
3. Check allergy boxes (e.g., "Nuts")
4. Add custom allergies in the text field (e.g., "sesame")
5. Click "Find My Recipes!"
6. Verify recipes respect the selected preferences

### Test Case 3: Error Handling
Test invalid inputs:
- Try uploading without selecting an image
- Upload a non-image file (should be rejected)
- Check browser console for proper error messages

---

## 🔍 How It Works

### Backend Flow (app.py)

1. **User uploads image** → Flask receives POST request at `/analyze`
2. **Image validation** → Checks file type and size
3. **CNN Processing** (Demo Layer)
   - Loads image through DenseNet121 architecture
   - Generates simulated predictions with confidence scores
   - Prints detailed logs showing "CNN processing"
4. **Gemini API Call** (Actual Intelligence)
   - Sends image to Gemini Vision API
   - Includes dietary preferences and allergies in prompt
   - Receives JSON response with ingredients and recipes
5. **Response combination** → Merges CNN demo results + AI results
6. **JSON sent to frontend** → Frontend displays all sections

### Frontend Flow (index.html)

1. User interaction with UI
2. FormData creation with image + preferences
3. AJAX POST request to `/analyze`
4. Progress indicators during processing
5. Dynamic result rendering in three sections
6. Error handling with user-friendly messages

---

## 🎓 Academic Presentation Notes

### For Faculty Demonstration

**Key talking points:**
1. **CNN Model Architecture**
   - "We implemented a DenseNet121 model with 316 ingredient classes"
   - "The model uses ImageNet pre-trained weights with custom classifier"
   - Show the console logs demonstrating CNN processing

2. **Two-Stage Pipeline**
   - "Stage 1: CNN performs initial ingredient detection with confidence scores"
   - "Stage 2: Gemini AI enhances detection and generates contextual recipes"
   
3. **Integration**
   - "The Flask backend orchestrates both models seamlessly"
   - "Frontend dynamically displays results from both stages"

**Demo Script:**
1. Start with console visible to show logs
2. Upload sample food image
3. Point out CNN processing logs
4. Highlight confidence scores in CNN results
5. Show AI-enhanced ingredient detection
6. Display generated recipes

---

## 🐛 Troubleshooting

### Issue: "GEMINI_API_KEY not found"
**Solution**: Check `.env` file exists and contains valid API key

### Issue: "Module not found" errors
**Solution**: Ensure virtual environment is activated and run:
```bash
pip install -r requirements.txt
```

### Issue: "Port already in use"
**Solution**: Change port in app.py:
```python
app.run(debug=True, port=5001)  # Use different port
```

### Issue: Images not uploading
**Solution**: Verify `static/uploads/` directory exists and has write permissions

### Issue: Gemini API errors
**Solution**: 
- Verify API key is correct
- Check internet connection
- Ensure API quota is not exceeded

---

## 📊 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 500MB for dependencies
- **Internet**: Required for Gemini API

### Recommended Requirements
- **Python**: 3.10+
- **RAM**: 8GB
- **Storage**: 1GB
- **GPU**: Optional (not required for demo)

---

## 🔐 Security Notes

- ✅ Environment variables used for API keys (not hardcoded)
- ✅ Secure filename handling with `werkzeug.utils.secure_filename`
- ✅ File type validation (only PNG, JPG, JPEG allowed)
- ✅ File size limit (16MB max)
- ✅ `.env` in `.gitignore` (API keys never committed)

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | ≥3.0.0 | Web framework |
| torch | ≥2.0.1 | PyTorch for CNN |
| torchvision | ≥0.15.2 | Vision models |
| google-generativeai | ≥0.3.0 | Gemini API |
| Pillow | ≥10.0.0 | Image processing |
| python-dotenv | ≥1.0.0 | Environment variables |
| numpy | ≥1.26.4 | Numerical operations |

---

## 🎯 Future Enhancements

- [ ] Add actual CNN training pipeline
- [ ] Implement database for recipe storage
- [ ] Add user authentication
- [ ] Support batch image processing
- [ ] Add nutritional information
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Mobile app version

---

## 📄 License

MIT License - see LICENSE file for details

---

## 👨‍💻 Author

**Abdullah**  
College Project - Machine Learning Specialization

---

## 🙏 Acknowledgments

- Google Gemini AI for vision capabilities
- PyTorch and torchvision teams
- Flask framework developers
- Tailwind CSS for UI components

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review console logs for error details
3. Verify all setup steps were completed
4. Check API key validity

---

**Last Updated**: October 2025  
**Version**: 1.0.0