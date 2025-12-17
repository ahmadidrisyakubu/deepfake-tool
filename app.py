from flask import Flask, request, render_template, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from flask_wtf import FlaskForm
from wtforms import FileField
from wtforms.validators import DataRequired
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from functools import wraps
import numpy as np
import tensorflow as tf
import random
import os
import time
import base64
import hashlib
import logging
import secrets
import html

# ===============================
# Deterministic inference
# ===============================
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# ===============================
# App initialization
# ===============================
app = Flask(__name__)

app.config.update(
    SECRET_KEY=secrets.token_hex(32),
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    WTF_CSRF_TIME_LIMIT=None,
    UPLOAD_FOLDER="uploads"
)

csrf = CSRFProtect(app)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per hour", "20 per minute"]
)
limiter.init_app(app)

# ===============================
# Logging
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("security.log"), logging.StreamHandler()]
)

# ===============================
# Load model
# ===============================
try:
    model = load_model("best_xception_model_finetuned.keras")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Model load failed: {e}")
    model = None

# ===============================
# Constants
# ===============================
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 10 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ===============================
# Helpers
# ===============================
def security_validate(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        logging.info(f"{request.remote_addr} → {request.endpoint}")
        return f(*args, **kwargs)
    return wrapper

class UploadForm(FlaskForm):
    image = FileField("Image", validators=[DataRequired()])

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    name, ext = os.path.splitext(secure_filename(filename))
    return f"{name}_{int(time.time())}{ext}"

def validate_file_security(file):
    errors = []

    if not file or file.filename == "":
        return ["No file provided"]

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)

    if size > MAX_FILE_SIZE:
        errors.append("File exceeds 10MB limit")

    if not allowed_file(file.filename):
        errors.append("Invalid file extension")

    try:
        from PIL import Image
        # Try to open as image to validate
        img = Image.open(file)
        img.verify()  # Verify it's a valid image
        file.seek(0)  # Reset file pointer
        
        # Check if it's JPEG or PNG by checking format
        if img.format not in ['JPEG', 'PNG', 'JPG']:
            errors.append(f"Invalid image format: {img.format}")
    except Exception as e:
        errors.append(f"Invalid image file: {str(e)}")
    finally:
        file.seek(0)

    return errors 

def generate_file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def predict_image(path):
    if not model:
        raise Exception("Model not loaded")

    img = image.load_img(path, target_size=(299, 299))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    pred = float(model.predict(arr, verbose=0)[0][0])
    label = "Real" if pred > 0.5 else "Fake"
    confidence = pred if pred > 0.5 else 1 - pred

    return label, round(confidence * 100, 2)

# ===============================
# Routes
# ===============================
@app.route("/", methods=["GET"])
@limiter.limit("30 per minute")
@security_validate
def index():
    return render_template("index.html", form=UploadForm())

@app.route("/predict", methods=["POST"])
@csrf.exempt
@limiter.limit("60 per minute")
@security_validate
def predict():
    file = request.files.get("image")

    errors = validate_file_security(file)
    if errors:
        return jsonify({"error": "; ".join(errors)}), 400

    filename = sanitize_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    try:
        label, confidence = predict_image(path)
        file_hash = generate_file_hash(path)

        logging.info(f"{filename} | {label} | {confidence}")

        return jsonify({
            "label": label,
            "confidence": confidence,
            "hash": file_hash
        })

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed"}), 500

    finally:
        if os.path.exists(path):
            os.remove(path)

# ===============================
# Security headers
# ===============================
@app.after_request
def security_headers(res):
    res.headers["X-Content-Type-Options"] = "nosniff"
    res.headers["X-Frame-Options"] = "DENY"
    res.headers["X-XSS-Protection"] = "1; mode=block"
    res.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    res.headers['Content-Security-Policy'] = (
    "default-src 'self' https://cdnjs.cloudflare.com; "
    "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
    "script-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "font-src 'self' https://cdnjs.cloudflare.com; "
    "frame-src https://www.youtube.com https://www.youtube-nocookie.com;"
    )

    return res

# ===============================
# Run
# ===============================
if __name__ == "__main__":
    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )
