"""
app.py — NeuraScan AI (v5)
Pipeline: YOLOv8 (no threshold) → RAG retrieval → Gemini LLM
"""

import os
import uuid
import time
import base64
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from detector import TumorDetector
from rag_engine import MedicalRAGEngine
from llm_agent import BrainTumorLLMAgent

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULTS_FOLDER"] = "results"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

os.makedirs("uploads",  exist_ok=True)
os.makedirs("results",  exist_ok=True)
os.makedirs("models",   exist_ok=True)

# ── Model Registry ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "brain_tumor": {
        "label":       "Brain Tumor (YOLOv26)",
        "model_path":  "models/brain_tumor_yolov26.pt",
        "classes":     ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "icon":        "🧠",
        "description": "Detects and classifies 4 brain tumor types from MRI scans",
        "available":   False,
    },
}

for key, meta in MODEL_REGISTRY.items():
    if os.path.exists(meta["model_path"]):
        meta["available"] = True
        print(f"[App] Model '{key}' found at {meta['model_path']}")
    else:
        print(f"[App] Model '{key}' NOT found — will run in demo mode")

detector  = TumorDetector(MODEL_REGISTRY)
rag       = MedicalRAGEngine()
llm_agent = BrainTumorLLMAgent(api_key=os.environ.get("GOOGLE_API_KEY"))

sessions: dict = {}


def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in {
        "png", "jpg", "jpeg", "bmp", "tiff", "webp"
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "yolo_models":   {k: v["available"] for k, v in MODEL_REGISTRY.items()},
        "llm_available": llm_agent.available,
        "llm_model":     f"Gemini ({llm_agent.available})",
    })


@app.route("/api/models")
def list_models():
    return jsonify({
        k: {
            "label":       v["label"],
            "icon":        v["icon"],
            "description": v["description"],
            "classes":     v["classes"],
            "available":   v["available"],
        }
        for k, v in MODEL_REGISTRY.items()
    })


@app.route("/api/session/new", methods=["POST"])
def new_session():
    sid = str(uuid.uuid4())
    sessions[sid] = {
        "id":               sid,
        "created_at":       datetime.utcnow().isoformat(),
        "detections":       [],
        "last_analysis":    "",
        "chat_history":     [],
        "last_image_path":  None,
        "last_model_label": "",
    }
    return jsonify({"session_id": sid})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image field"}), 400
    f = request.files["image"]
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = f"{uuid.uuid4()}_{secure_filename(f.filename)}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(filepath)

    return jsonify({
        "success":  True,
        "filename": filename,
        "original": f.filename,
        "url":      f"/uploads/{filename}",
    })


@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Step 1: Run YOLOv8 (no threshold) + RAG retrieval.
    Confidence slider removed — model decides everything.
    """
    data       = request.json or {}
    filename   = data.get("filename")
    model_key  = data.get("model", "brain_tumor")
    session_id = data.get("session_id")

    if not filename:
        return jsonify({"error": "filename required"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found on server"}), 404

    meta = MODEL_REGISTRY.get(model_key)
    if not meta:
        return jsonify({"error": "Unknown model"}), 400

    t0 = time.time()

    # ── Run detection — no conf parameter ─────────────────────────
    if meta["available"]:
        result = detector.run(filepath, model_key)          # real YOLO
    else:
        result = detector.demo_result(filepath, model_key)  # demo fallback

    elapsed = round(time.time() - t0, 3)

    # ── RAG: use detected class names for retrieval ────────────────
    detected_classes = list({d["class"] for d in result["detections"]})
    rag_context      = rag.get_context(detected_classes, model_key)

    # ── Save annotated result image ───────────────────────────────
    result_filename = f"result_{filename}"
    result_path     = os.path.join(app.config["RESULTS_FOLDER"], result_filename)
    if result.get("annotated_image_b64"):
        with open(result_path, "wb") as out:
            out.write(base64.b64decode(result["annotated_image_b64"]))

    # ── Persist in session ────────────────────────────────────────
    if session_id and session_id in sessions:
        sessions[session_id]["detections"]       = result["detections"]
        sessions[session_id]["last_image_path"]  = filepath
        sessions[session_id]["last_model_label"] = meta["label"]
        sessions[session_id]["last_analysis"]    = ""

    return jsonify({
        "success":       True,
        "model":         model_key,
        "model_label":   meta["label"],
        "demo_mode":     not meta["available"],
        "elapsed_ms":    int(elapsed * 1000),
        "detections":    result["detections"],
        "summary":       result["summary"],
        "annotated_url": f"/results/{result_filename}" if result.get("annotated_image_b64") else None,
        "rag_context":   rag_context,
        "classes":       meta["classes"],
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Step 2: Gemini reads the image + YOLO detections → full clinical report."""
    data        = request.json or {}
    filename    = data.get("filename")
    detections  = data.get("detections", [])
    rag_context = data.get("rag_context", [])
    model_label = data.get("model_label", "Brain Tumor YOLOv8")
    session_id  = data.get("session_id")

    if not filename:
        return jsonify({"error": "filename required"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "Image not found"}), 404

    result = llm_agent.analyze(
        image_path=filepath,
        detections=detections,
        rag_context=rag_context,
        model_label=model_label,
    )

    if session_id and session_id in sessions:
        sessions[session_id]["last_analysis"] = result.get("report", "")

    return jsonify({
        "success":   True,
        "report":    result["report"],
        "demo_mode": result["demo_mode"],
    })


@app.route("/api/chat", methods=["POST"])
def chat():

    data       = request.json or {}

    question   = data.get("question", "").strip()

    session_id = data.get("session_id")


    if not question:

        return jsonify({"error": "question required"}), 400


    sess = sessions.get(session_id, {})


    detections       = sess.get("detections", [])

    analysis_summary = sess.get("last_analysis", "")

    model_label      = sess.get("last_model_label", "Brain Tumor YOLOv8")

    image_path       = sess.get("last_image_path")


    if session_id in sessions:

        sessions[session_id]["chat_history"].append({

            "role": "user",

            "content": question

        })


    try:

        response = llm_agent.chat(

            question=question,

            detections=detections,

            analysis_summary=analysis_summary,

            model_label=model_label,

            image_path=image_path,

        )


        if not isinstance(response, str):

            response = str(response)


    except Exception as e:

        response = f"⚠️ Error: {str(e)}"


    if session_id in sessions:

        sessions[session_id]["chat_history"].append({

            "role": "assistant",

            "content": response

        })


    return jsonify({

        "success": True,

        "response": response

    })


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
