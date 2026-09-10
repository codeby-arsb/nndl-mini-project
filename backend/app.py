import os
import sys
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
SAMPLES_DIR = os.path.join(FRONTEND_DIR, "assets", "samples")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

from backend.inference_service import ColorizationInferenceService

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# Initialize isolated inference service (loads model once on startup)
inference_service = ColorizationInferenceService()

@app.route("/")
def index():
    """Serve main frontend page."""
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    """Serve frontend static assets (CSS, JS, images)."""
    return send_from_directory(FRONTEND_DIR, path)

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "ColorAI Image Colorization API",
        "version": "1.1.0"
    })

@app.route("/api/info", methods=["GET"])
def model_info():
    """Return model architecture and runtime details."""
    try:
        info = inference_service.get_system_info()
        return jsonify({
            "success": True,
            "data": info
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Unable to retrieve model runtime information."
        }), 500

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return list of preloaded sample images for 1-click viva demonstrations."""
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for fname in sorted(os.listdir(SAMPLES_DIR)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                sample_id = os.path.splitext(fname)[0]
                label = sample_id.replace("-", " ").replace("_", " ").title()
                samples.append({
                    "id": sample_id,
                    "filename": fname,
                    "title": label,
                    "url": f"/assets/samples/{fname}"
                })
    return jsonify({
        "success": True,
        "samples": samples
    })

@app.route("/api/upload_checkpoint", methods=["POST"])
def upload_checkpoint():
    """
    Allows user or group to upload a trained .pth checkpoint file (e.g. best.pth).
    Saves to outputs/checkpoints/best.pth and hot-reloads the weights into the running model.
    """
    if "checkpoint" not in request.files and "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No checkpoint file provided. Please select a .pth model file."
        }), 400

    chk_file = request.files.get("checkpoint") or request.files.get("file")
    fname = chk_file.filename or "best.pth"
    if not fname.lower().endswith((".pth", ".pt")):
        return jsonify({
            "success": False,
            "error": "Invalid file format. Model checkpoints must end with .pth or .pt"
        }), 400

    target_path = os.path.join(CHECKPOINTS_DIR, "best.pth")
    try:
        chk_file.save(target_path)
        success = inference_service.load_checkpoint(target_path)
        if success:
            info = inference_service.get_system_info()
            return jsonify({
                "success": True,
                "message": f"Successfully loaded checkpoint: {fname}",
                "data": info
            })
        else:
            return jsonify({
                "success": False,
                "error": "The uploaded file could not be parsed as a valid PyTorch model state dictionary."
            }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to save checkpoint: {str(e)}"
        }), 500

@app.route("/api/colorize", methods=["POST"])
def colorize():
    """
    Colorize an uploaded image or preloaded sample image.
    Accepts:
      - multipart/form-data with file field 'image' or 'file'
      - application/json with 'image' (base64 string) or 'sample_id' (name of sample file)
    Optional parameters:
      - mode: 'hd' (default, native resolution) or 'model' (256x256)
      - saturation: float (0.0 to 2.0, default 1.0)
      - tint: float (-50 to 50, default 0.0) -> a* shift
      - warmth: float (-50 to 50, default 0.0) -> b* shift
      - denoise: int (0 to 10, default 0)
      - auto_balance: bool (default True)
    """
    image_bytes = None
    params = {
        "engine": "pretrained",
        "mode": "hd",
        "saturation": 1.0,
        "tint": 0.0,
        "warmth": 0.0,
        "denoise": 0,
        "auto_balance": True
    }
    
    # 1. Check multipart/form-data upload
    if request.files:
        if "image" in request.files:
            uploaded_file = request.files["image"]
            image_bytes = uploaded_file.read()
        elif "file" in request.files:
            uploaded_file = request.files["file"]
            image_bytes = uploaded_file.read()
            
        # Parse form options if present
        if "engine" in request.form:
            params["engine"] = request.form.get("engine", "pretrained")
        if "mode" in request.form:
            params["mode"] = request.form.get("mode", "hd")
        if "saturation" in request.form:
            params["saturation"] = float(request.form.get("saturation", 1.0))
        if "tint" in request.form:
            params["tint"] = float(request.form.get("tint", 0.0))
        if "warmth" in request.form:
            params["warmth"] = float(request.form.get("warmth", 0.0))
        if "denoise" in request.form:
            params["denoise"] = int(request.form.get("denoise", 0))
        if "auto_balance" in request.form:
            params["auto_balance"] = request.form.get("auto_balance", "true").lower() in ("true", "1")
        
    # 2. Check JSON payload
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        
        for k in params.keys():
            if k in data:
                params[k] = data[k]
        
        # Option A: sample_id
        if "sample_id" in data:
            sample_id = data["sample_id"]
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                candidate = os.path.join(SAMPLES_DIR, f"{sample_id}{ext}")
                if os.path.exists(candidate):
                    with open(candidate, "rb") as f:
                        image_bytes = f.read()
                    break
            if image_bytes is None:
                return jsonify({
                    "success": False,
                    "error": f"Sample image '{sample_id}' not found."
                }), 404
                
        # Option B: base64 encoded data
        elif "image" in data:
            raw_b64 = data["image"]
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(raw_b64)
            except Exception:
                return jsonify({
                    "success": False,
                    "error": "Failed to decode base64 image data."
                }), 400

    if not image_bytes:
        return jsonify({
            "success": False,
            "error": "No image provided. Please upload an image file (JPG, PNG, WEBP) or select a demo sample."
        }), 400

    # Max file size limit: 15 MB
    if len(image_bytes) > 15 * 1024 * 1024:
        return jsonify({
            "success": False,
            "error": "File size exceeds the 15MB limit. Please upload a smaller image."
        }), 413

    # Run inference through upgraded service
    try:
        result = inference_service.colorize_image(
            image_bytes,
            engine=params.get("engine", "pretrained"),
            mode=params.get("mode", "hd"),
            saturation=float(params.get("saturation", 1.0)),
            tint=float(params.get("tint", 0.0)),
            warmth=float(params.get("warmth", 0.0)),
            denoise=int(params.get("denoise", 0)),
            auto_balance=bool(params.get("auto_balance", True))
        )
        return jsonify(result)
    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400
    except Exception as e:
        print(f"[ColorAI API Error] Inference failure: {e}", file=sys.stderr)
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while colorizing the image. Please verify the image file format and try again."
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting ColorAI Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
