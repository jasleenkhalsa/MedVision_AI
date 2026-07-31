"""
app.py — MedVision AI

Pipeline:
MRI Upload
→ YOLO26 Detection
→ Metadata and Urgency Generation
→ FAISS RAG Retrieval
→ Qwen2.5 Clinical Report
"""

import base64
import os
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from detector import TumorDetector
from llm_agent import BrainTumorLLMAgent
from metadata_engine import MetadataEngine
from rag_engine import MedicalRAGEngine


# ──────────────────────────────────────────────────────────────────────────────
# Application configuration
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULTS_FOLDER"] = "results"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)
os.makedirs("models", exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "brain_tumor": {
        "label": "Brain Tumor (YOLOv26)",
        "model_path": "models/brain_tumor_yolo26.pt",
        "classes": [
            "Glioma",
            "Meningioma",
            "No Tumor",
            "Pituitary",
        ],
        "icon": "🧠",
        "description": (
            "Detects and classifies four brain-tumor categories "
            "from MRI scans."
        ),
        "available": False,
    },
}


# Check whether each registered model exists.
for model_key, model_metadata in MODEL_REGISTRY.items():
    model_path = model_metadata["model_path"]

    if os.path.exists(model_path):
        model_metadata["available"] = True
        print(
            f"[App] Model '{model_key}' found at "
            f"{model_path}"
        )
    else:
        model_metadata["available"] = False
        print(
            f"[App] Model '{model_key}' NOT found — "
            "demo mode will be used"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Application services
# ──────────────────────────────────────────────────────────────────────────────

detector = TumorDetector(MODEL_REGISTRY)
rag = MedicalRAGEngine()
metadata_engine = MetadataEngine()

llm_agent = BrainTumorLLMAgent(
    api_key=os.environ.get("GOOGLE_API_KEY")
)


# Temporary in-memory sessions.
# These sessions are deleted whenever the Flask server restarts.
sessions: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    """Return True when the uploaded image extension is supported."""

    allowed_extensions = {
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tiff",
        "webp",
    }

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )


# ──────────────────────────────────────────────────────────────────────────────
# Web page
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the main application page."""

    return render_template("index.html")


# ──────────────────────────────────────────────────────────────────────────────
# Status and model routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/status")
def status():
    """Return the availability of the detector and local LLM."""

    return jsonify({
        "yolo_models": {
            key: metadata["available"]
            for key, metadata in MODEL_REGISTRY.items()
        },
        "llm_available": llm_agent.available,
        "llm_model": (
            "Qwen2.5:1.5b (Ollama)"
            if llm_agent.available
            else "Unavailable"
        ),
        "rag_engine": "SentenceTransformer + FAISS",
    })


@app.route("/api/models")
def list_models():
    """Return information about all registered detection models."""

    return jsonify({
        key: {
            "label": metadata["label"],
            "icon": metadata["icon"],
            "description": metadata["description"],
            "classes": metadata["classes"],
            "available": metadata["available"],
        }
        for key, metadata in MODEL_REGISTRY.items()
    })


# ──────────────────────────────────────────────────────────────────────────────
# Session route
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/session/new", methods=["POST"])
def new_session():
    """Create a temporary analysis session."""

    session_id = str(uuid.uuid4())

    sessions[session_id] = {
        "id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detections": [],
        "metadata": {},
        "rag_context": [],
        "last_analysis": "",
        "chat_history": [],
        "last_image_path": None,
        "last_model_label": "",
    }

    return jsonify({
        "success": True,
        "session_id": session_id,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Upload route
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload():
    """Save an uploaded MRI image."""

    if "image" not in request.files:
        return jsonify({
            "error": "No image field was provided."
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({
            "error": "No file was selected."
        }), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({
            "error": "Unsupported file type."
        }), 400

    original_filename = secure_filename(uploaded_file.filename)

    stored_filename = (
        f"{uuid.uuid4()}_{original_filename}"
    )

    stored_filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        stored_filename,
    )

    uploaded_file.save(stored_filepath)

    return jsonify({
        "success": True,
        "filename": stored_filename,
        "original": uploaded_file.filename,
        "url": f"/uploads/{stored_filename}",
    })


# ──────────────────────────────────────────────────────────────────────────────
# Detection route
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Run YOLO26 detection, generate structured metadata and retrieve
    relevant clinical passages using SentenceTransformer and FAISS.
    """

    data = request.get_json(silent=True) or {}

    filename = data.get("filename")
    model_key = data.get("model", "brain_tumor")
    session_id = data.get("session_id")

    if not filename:
        return jsonify({
            "error": "filename required"
        }), 400

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename,
    )

    if not os.path.exists(filepath):
        return jsonify({
            "error": "File not found on server"
        }), 404

    model_metadata = MODEL_REGISTRY.get(model_key)

    if not model_metadata:
        return jsonify({
            "error": "Unknown model"
        }), 400

    start_time = time.time()

    try:
        if model_metadata["available"]:
            detection_result = detector.run(
                filepath,
                model_key,
            )
        else:
            detection_result = detector.demo_result(
                filepath,
                model_key,
            )

    except Exception as error:
        print(f"[Detection Error] {error}")

        return jsonify({
            "error": "Detection failed",
            "details": str(error),
        }), 500

    elapsed_seconds = time.time() - start_time
    elapsed_ms = int(elapsed_seconds * 1000)

    detections = detection_result.get("detections", [])

    # Generate deterministic metadata and urgency.
    generated_metadata = metadata_engine.generate(
        image_path=filepath,
        detections=detections,
        model_label=model_metadata["label"],
        elapsed_ms=elapsed_ms,
    )

    # Retrieve relevant evidence using the detected class names.
    detected_classes = list({
        detection.get("class")
        for detection in detections
        if detection.get("class")
    })

    rag_context = rag.get_context(
        detected_classes=detected_classes,
        model_key=model_key,
    )

    # Save the annotated image.
    result_filename = f"result_{filename}"

    result_path = os.path.join(
        app.config["RESULTS_FOLDER"],
        result_filename,
    )

    annotated_image_b64 = detection_result.get(
        "annotated_image_b64"
    )

    annotated_url = None

    if annotated_image_b64:
        try:
            with open(result_path, "wb") as output_file:
                output_file.write(
                    base64.b64decode(annotated_image_b64)
                )

            annotated_url = f"/results/{result_filename}"

        except Exception as error:
            print(
                "[App] Could not save annotated image: "
                f"{error}"
            )

    # Save information in the current session.
    if session_id and session_id in sessions:
        sessions[session_id]["detections"] = detections
        sessions[session_id]["metadata"] = generated_metadata
        sessions[session_id]["rag_context"] = rag_context
        sessions[session_id]["last_image_path"] = filepath
        sessions[session_id]["last_model_label"] = (
            model_metadata["label"]
        )
        sessions[session_id]["last_analysis"] = ""

    return jsonify({
        "success": True,
        "model": model_key,
        "model_label": model_metadata["label"],
        "demo_mode": not model_metadata["available"],
        "elapsed_ms": elapsed_ms,
        "detections": detections,
        "summary": detection_result.get("summary", {}),
        "metadata": generated_metadata,
        "annotated_url": annotated_url,
        "rag_context": rag_context,
        "classes": model_metadata["classes"],
    })


# ──────────────────────────────────────────────────────────────────────────────
# Clinical report route
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Generate a structured clinical decision-support report using
    YOLO detections, deterministic metadata and retrieved RAG evidence.
    """

    data = request.get_json(silent=True) or {}

    filename = data.get("filename")
    detections = data.get("detections", [])
    rag_context = data.get("rag_context", [])
    generated_metadata = data.get("metadata", {})
    model_label = data.get(
        "model_label",
        "Brain Tumor (YOLOv26)",
    )
    session_id = data.get("session_id")

    # Use saved session information when the frontend does not send it.
    if session_id and session_id in sessions:
        current_session = sessions[session_id]

        if not detections:
            detections = current_session.get(
                "detections",
                [],
            )

        if not generated_metadata:
            generated_metadata = current_session.get(
                "metadata",
                {},
            )

        if not rag_context:
            rag_context = current_session.get(
                "rag_context",
                [],
            )

        if model_label == "Brain Tumor (YOLOv26)":
            model_label = current_session.get(
                "last_model_label",
                model_label,
            )

    if not filename:
        return jsonify({
            "error": "filename required"
        }), 400

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename,
    )

    if not os.path.exists(filepath):
        return jsonify({
            "error": "Image not found"
        }), 404

    try:
        result = llm_agent.analyze(
            image_path=filepath,
            detections=detections,
            rag_context=rag_context,
            metadata=generated_metadata,
            model_label=model_label,
        )

    except TypeError as error:
        print(f"[LLM Signature Error] {error}")

        return jsonify({
            "error": (
                "The analyze() method in llm_agent.py does not yet "
                "accept the metadata parameter."
            ),
            "details": str(error),
        }), 500

    except Exception as error:
        print(f"[LLM Analysis Error] {error}")

        return jsonify({
            "error": "Clinical report generation failed",
            "details": str(error),
        }), 500

    report = result.get(
        "report",
        "No report was generated.",
    )

    if session_id and session_id in sessions:
        sessions[session_id]["last_analysis"] = report

    return jsonify({
        "success": True,
        "report": report,
        "demo_mode": result.get("demo_mode", False),
        "metadata": generated_metadata,
        "rag_context": rag_context,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Follow-up chat route
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    """Answer follow-up questions about the latest analysis."""

    data = request.get_json(silent=True) or {}

    question = data.get("question", "").strip()
    session_id = data.get("session_id")

    if not question:
        return jsonify({
            "error": "question required"
        }), 400

    current_session = sessions.get(session_id, {})

    detections = current_session.get(
        "detections",
        [],
    )

    generated_metadata = current_session.get(
        "metadata",
        {},
    )

    analysis_summary = current_session.get(
        "last_analysis",
        "",
    )

    model_label = current_session.get(
        "last_model_label",
        "Brain Tumor (YOLOv26)",
    )

    image_path = current_session.get(
        "last_image_path"
    )

    if session_id in sessions:
        sessions[session_id]["chat_history"].append({
            "role": "user",
            "content": question,
        })

    try:
        # Keep the existing chat() signature unless you also update
        # llm_agent.py to accept metadata.
        response = llm_agent.chat(
            question=question,
            detections=detections,
            analysis_summary=analysis_summary,
            model_label=model_label,
            image_path=image_path,
        )

        if not isinstance(response, str):
            response = str(response)

    except Exception as error:
        print(f"[Chat Error] {error}")
        response = f"⚠️ Error: {error}"

    if session_id in sessions:
        sessions[session_id]["chat_history"].append({
            "role": "assistant",
            "content": response,
        })

    return jsonify({
        "success": True,
        "response": response,
        "metadata": generated_metadata,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Static image routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Serve an uploaded MRI image."""

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
    )


@app.route("/results/<path:filename>")
def serve_result(filename):
    """Serve an annotated detection result."""

    return send_from_directory(
        app.config["RESULTS_FOLDER"],
        filename,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Application entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False,
        host="0.0.0.0",
        port=5000,
    )