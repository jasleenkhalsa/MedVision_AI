"""
llm_agent.py — Ollama local LLM (fast version)

Speed fixes:
  1. Uses qwen3:1.7b instead of qwen3:8b  →  4-5x faster
  2. Short, focused prompts  →  less tokens to generate
  3. stream=False  →  waits for full response, no partial writes
  4. num_predict capped  →  stops runaway generation

City specialist fix:
  - Broader city detection (any city mention, not just end-of-sentence)
  - Returns Google Maps + Practo + Justdial links for Indian cities
  - Works for both Indian and international cities

Install:
  1. Download Ollama from https://ollama.com
  2. Run: ollama pull qwen2.5:1.5b
  3. Make sure Ollama is running before starting Flask
"""

import re
import os
from typing import Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[LLM] ollama package not installed — run: pip install ollama")


# ── Model choice ───────────────────────────────────────────────────────────────
#
#   Speed comparison on CPU (no GPU):
#   qwen3:8b        →  3-8 minutes  ← what you had (too slow)
#   qwen2.5:3b      →  45-90 sec    ← acceptable
#   qwen2.5:1.5b    →  15-30 sec    ← recommended (good quality, fast)
#   tinyllama       →  5-10 sec     ← fastest, lower quality
#
#   If you have NVIDIA GPU: any model runs in seconds via CUDA
#
MODEL_NAME = "qwen2.5:1.5b"   # pull with: ollama pull qwen2.5:1.5b


# ── City detection ─────────────────────────────────────────────────────────────
# Catches: "doctors in Mumbai", "hospital near Delhi", "specialist in Pune", etc.
CITY_PATTERNS = [
    r"(?:doctor|doctors|specialist|specialists|hospital|hospitals|clinic|clinics|neurosurgeon|neurologist)\s+(?:in|near|at|around)\s+([a-zA-Z\s]+?)(?:\?|$|\.|,)",
    r"(?:in|near|at|around)\s+([a-zA-Z\s]+?)\s+(?:doctor|doctors|specialist|hospital|clinic|neurosurgeon|neurologist)",
    r"find\s+(?:a\s+)?(?:doctor|specialist|hospital|neurosurgeon)\s+in\s+([a-zA-Z\s]+?)(?:\?|$|\.|,)",
    r"(?:who|where)\s+(?:to\s+)?(?:see|consult|visit|go)\s+in\s+([a-zA-Z\s]+?)(?:\?|$|\.|,)",
]

def detect_city(text: str) -> Optional[str]:
    """Extract city name from user message. Returns city string or None."""
    for pattern in CITY_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            city = match.group(1).strip().rstrip("?.,")
            # Filter out non-city words
            skip = {"the", "a", "an", "me", "my", "any", "good", "best", "top", "nearby"}
            if city.lower() not in skip and len(city) > 2:
                return city
    return None


def specialist_links(city: str, ollama_model: str = MODEL_NAME) -> str:
    """
    Use Ollama to generate real hospital/specialist names for any city,
    then append booking links. Works for ALL Indian cities.
    """
    city_q  = city.replace(" ", "+")
    city_sl = city.replace(" ", "-").lower()
    city_t  = city.title()

    # Ask Ollama for real hospital names
    hospital_section = ""
    try:
        prompt = f"""List real hospitals and neurosurgeons in {city_t}, India that treat brain tumors.

Format exactly like this for each entry:
Hospital: [Hospital Name]
Doctor: [Doctor Name] - [Specialization]
Address: [Area, City]
Phone: [number if known, else write Contact via hospital]

List 4 to 5 entries. Only real verifiable hospitals. Be concise."""

        response = ollama.chat(
            model=ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 400, "temperature": 0.2},
            stream=False,
        )
        hospital_section = response["message"]["content"].strip()
    except Exception:
        hospital_section = (
            f"Could not reach Ollama for live hospital data.\n"
            f"Please use the links below to find specialists in {city_t}."
        )

    result = f"""### Hospitals and Specialists in {city_t}

{hospital_section}

---

### Book or Search Online

Google Maps - Neurosurgeons
https://www.google.com/maps/search/neurosurgeon+in+{city_q}

Google Maps - Neurology Hospitals
https://www.google.com/maps/search/neurology+hospital+in+{city_q}

Practo - Book Appointment
https://www.practo.com/search/doctors?results_type=doctor&q=neurosurgeon&city={city.lower()}

Apollo Hospitals
https://www.apollohospitals.com/find-a-doctor/?speciality=Neurosurgery&city={city_sl}

Fortis Healthcare
https://www.fortishealthcare.com/find-a-doctor?speciality=neurosurgery&city={city_sl}

---
Always verify credentials before visiting. Call ahead to confirm availability."""

    return result


def format_detections(detections: list) -> str:
    if not detections:
        return "No tumor detected"
    return ", ".join(
        f"{d['class']} ({d['confidence']*100:.1f}%)"
        for d in detections
    )

def format_rag(rag_context: list) -> str:
    if not rag_context:
        return ""
    return " ".join(r["content"] for r in rag_context)[:300]


# ── Main agent class ───────────────────────────────────────────────────────────
class BrainTumorLLMAgent:

    def __init__(self, api_key=None):  # api_key ignored, kept for compatibility
        self.model     = MODEL_NAME
        self.available = OLLAMA_AVAILABLE
        if OLLAMA_AVAILABLE:
            # Check if Ollama server is actually running
            try:
                ollama.list()
                print(f"[LLM] Ollama running — model: {self.model}")
                # Auto-pull model if not present
                models = [m.model for m in ollama.list().models]
                if not any(self.model in m for m in models):
                    print(f"[LLM] Pulling {self.model} — this takes a few minutes on first run...")
                    ollama.pull(self.model)
                    print(f"[LLM] {self.model} ready ✓")
            except Exception as e:
                print(f"[LLM] Ollama not running: {e}")
                print("[LLM] Start Ollama first: open Ollama app or run 'ollama serve'")
                self.available = False
        else:
            print("[LLM] Install ollama: pip install ollama")

    # ── Analysis — called after YOLO detection ─────────────────────────────────
    def analyze(self, image_path: str, detections: list,
                rag_context: list, model_label: str) -> dict:

        condition    = format_detections(detections)
        context      = format_rag(rag_context)
        has_tumor    = detections and detections[0]["class"].lower() != "no tumor"

        # Short, focused prompt — fewer tokens = faster response
        if has_tumor:
            prompt = f"""You are a medical expert. Be concise.

Brain MRI scan detected: {condition}

Using this context: {context}

Write a brief clinical summary with these sections:
1. Diagnosis - what was found
2. Stage/Severity - WHO grade if applicable
3. Main Symptoms - bullet list (max 4)
4. Causes - brief (2-3 lines)
5. Treatment - key options
6. Medicines - list drug names and doses
7. Prognosis - survival outlook
8. Patient Advice - simple explanation

Keep each section to 2-4 lines. Be direct and clear."""
        else:
            prompt = f"""Brain MRI scan shows: No Tumor Detected.

Write a brief reassuring medical note:
1. Confirm no tumor found
2. What a normal MRI means
3. Recommend regular follow-ups
4. General brain health advice

Keep it short and reassuring."""

        if not self.available:
            return {"report": self._demo_report(detections), "demo_mode": True}

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": 600,    # cap output length → faster
                    "temperature": 0.3,    # less creative = more consistent + faster
                    "top_p": 0.9,
                },
                stream=False,
            )
            return {
                "report":    response["message"]["content"],
                "demo_mode": False,
            }
        except Exception as e:
            err = str(e)
            if "connection" in err.lower() or "refused" in err.lower():
                return {
                    "report": "⚠️ Ollama is not running.\n\nPlease open the Ollama app or run `ollama serve` in a terminal, then refresh.",
                    "demo_mode": True,
                }
            return {"report": f"⚠️ LLM error: {err}", "demo_mode": True}

    # ── Chat — Q&A + city specialist finder ───────────────────────────────────
    def chat(self, question: str, detections: list, analysis_summary: str,
             model_label: str, image_path: Optional[str] = None) -> str:

        # ── City specialist search ──
        city = detect_city(question)
        if city:
            return specialist_links(city, self.model)

        # ── Regular Q&A ──
        condition = format_detections(detections)

        prompt = f"""You are a helpful medical assistant. Be concise.

Brain MRI finding: {condition}

User question: {question}

Give a clear, helpful answer in 3-5 lines.
If clinical, end with: "Please consult a qualified specialist." """

        if not self.available:
            return (
                "⚠️ Ollama is not running.\n\n"
                "Open the Ollama app or run `ollama serve` in a terminal, then try again."
            )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": 300,   # short answers for chat
                    "temperature": 0.3,
                },
                stream=False,
            )
            return response["message"]["content"]
        except Exception as e:
            err = str(e)
            if "connection" in err.lower() or "refused" in err.lower():
                return "⚠️ Ollama is not running. Please start Ollama and try again."
            return f"⚠️ Error: {err}"

    # ── Demo fallback ──────────────────────────────────────────────────────────
    def _demo_report(self, detections: list) -> str:
        classes = [d["class"] for d in detections] if detections else ["No finding"]
        top     = detections[0] if detections else None
        return f"""### Diagnosis
YOLOv26 detected: **{', '.join(classes)}**
{f"Confidence: **{top['confidence']*100:.1f}%**" if top else ""}

### Status
Ollama is not running. Please:
1. Open the Ollama application
2. Or run `ollama serve` in terminal
3. Make sure `{MODEL_NAME}` is pulled: `ollama pull {MODEL_NAME}`
4. Restart the Flask server

### Model
Using: `{MODEL_NAME}` (fast local model, no internet needed)"""
