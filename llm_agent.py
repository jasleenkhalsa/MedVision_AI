"""
llm_agent.py — MedVision AI local LLM agent

Pipeline:
YOLO26 detections
→ deterministic metadata and urgency
→ FAISS-retrieved clinical evidence
→ Qwen2.5 report through Ollama

Important:
This module generates clinical decision-support text only.
It does not provide a confirmed diagnosis, medication prescription,
WHO grading, survival prediction, or replacement for radiological review.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional
from urllib.parse import quote_plus

try:
    import ollama

    OLLAMA_AVAILABLE = True

except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False

    print(
        "[LLM] Ollama package is not installed. "
        "Run: pip install ollama"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Ollama configuration
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "qwen2.5:1.5b"


# ──────────────────────────────────────────────────────────────────────────────
# City detection
# ──────────────────────────────────────────────────────────────────────────────

CITY_PATTERNS = [
    (
        r"(?:doctor|doctors|specialist|specialists|hospital|hospitals|"
        r"clinic|clinics|neurosurgeon|neurologist)\s+"
        r"(?:in|near|at|around)\s+"
        r"([a-zA-Z\s]+?)(?:\?|$|\.|,)"
    ),
    (
        r"(?:in|near|at|around)\s+"
        r"([a-zA-Z\s]+?)\s+"
        r"(?:doctor|doctors|specialist|hospital|clinic|"
        r"neurosurgeon|neurologist)"
    ),
    (
        r"find\s+(?:a\s+)?"
        r"(?:doctor|specialist|hospital|neurosurgeon)\s+in\s+"
        r"([a-zA-Z\s]+?)(?:\?|$|\.|,)"
    ),
    (
        r"(?:who|where)\s+(?:to\s+)?"
        r"(?:see|consult|visit|go)\s+in\s+"
        r"([a-zA-Z\s]+?)(?:\?|$|\.|,)"
    ),
]


def detect_city(text: str) -> Optional[str]:
    """Extract a possible city name from a user question."""

    ignored_values = {
        "the",
        "a",
        "an",
        "me",
        "my",
        "any",
        "good",
        "best",
        "top",
        "nearby",
    }

    for pattern in CITY_PATTERNS:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        city = match.group(1).strip().rstrip("?.,")
        city = " ".join(city.split())

        if (
            len(city) > 2
            and city.lower() not in ignored_values
        ):
            return city

    return None


def specialist_links(city: str) -> str:
    """
    Return search links for hospitals and specialists.

    The LLM is intentionally not asked to generate doctor or hospital names,
    because a local language model can produce inaccurate or outdated details.
    """

    city_name = city.title()
    encoded_city = quote_plus(city_name)

    google_neurosurgeon = (
        "https://www.google.com/maps/search/"
        f"neurosurgeon+in+{encoded_city}"
    )

    google_neurology_hospital = (
        "https://www.google.com/maps/search/"
        f"neurology+hospital+in+{encoded_city}"
    )

    practo_link = (
        "https://www.practo.com/search/doctors"
        "?results_type=doctor"
        "&q=neurosurgeon"
        f"&city={encoded_city}"
    )

    return f"""
### Specialist search for {city_name}

Use these services to search for qualified neurosurgeons,
neurologists, radiologists, or brain-tumor care centres:

- Google Maps — Neurosurgeons:
  {google_neurosurgeon}

- Google Maps — Neurology hospitals:
  {google_neurology_hospital}

- Practo — Neurosurgeon appointments:
  {practo_link}

Before booking:

1. Verify the clinician's qualification and hospital affiliation.
2. Confirm that the specialist treats brain or pituitary conditions.
3. Carry the complete MRI study and formal radiology report.
4. Contact emergency services for severe headache, sudden weakness,
   seizures, loss of consciousness, or sudden visual loss.

The links are search tools, not medical endorsements.
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

def format_detections(
    detections: list[dict[str, Any]],
) -> str:
    """Convert detector results into readable text."""

    if not detections:
        return "No detector bounding boxes were returned."

    formatted_items: list[str] = []

    for detection in detections:
        class_name = detection.get(
            "class",
            "Unknown finding",
        )

        confidence = float(
            detection.get("confidence", 0)
        )

        bbox = detection.get("bbox")

        item = (
            f"Class: {class_name}; "
            f"confidence: {confidence * 100:.2f}%"
        )

        if bbox:
            item += f"; bounding box: {bbox}"

        formatted_items.append(item)

    return "\n".join(formatted_items)


def format_metadata(
    metadata: dict[str, Any],
) -> str:
    """Convert deterministic metadata into formatted JSON."""

    if not metadata:
        return "No structured metadata was supplied."

    return json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def format_rag(
    rag_context: list[dict[str, Any]],
) -> str:
    """
    Format FAISS retrieval results.

    The new rag_engine.py returns:
    id, category, source, text and score.
    """

    if not rag_context:
        return (
            "No external knowledge passages were retrieved. "
            "Use only the supplied detection and metadata."
        )

    passages: list[str] = []

    for index, item in enumerate(
        rag_context,
        start=1,
    ):
        source = item.get(
            "source",
            "Clinical knowledge base",
        )

        category = item.get(
            "category",
            "General",
        )

        score = item.get(
            "score",
            "Not provided",
        )

        # Support both the new "text" key and the old "content" key.
        text = item.get(
            "text",
            item.get("content", ""),
        )

        if not text:
            continue

        passages.append(
            (
                f"Evidence {index}\n"
                f"Source: {source}\n"
                f"Category: {category}\n"
                f"Similarity score: {score}\n"
                f"Passage: {text}"
            )
        )

    if not passages:
        return "No usable RAG passages were supplied."

    return "\n\n".join(passages)


def has_tumor_detection(
    detections: list[dict[str, Any]],
) -> bool:
    """Return True when at least one detection is not the No Tumor class."""

    for detection in detections:
        class_name = str(
            detection.get("class", "")
        ).strip().lower()

        if class_name and class_name != "no tumor":
            return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# Main LLM agent
# ──────────────────────────────────────────────────────────────────────────────

class BrainTumorLLMAgent:
    """Generate local reports and follow-up answers through Ollama."""

    def __init__(
        self,
        api_key: Optional[str] = None,
    ) -> None:
        # api_key is retained only for compatibility with the existing app.py.
        _ = api_key

        self.model = MODEL_NAME
        self.available = OLLAMA_AVAILABLE

        if not OLLAMA_AVAILABLE:
            print(
                "[LLM] Install the Ollama Python package with: "
                "pip install ollama"
            )
            return

        try:
            model_response = ollama.list()

            print(
                f"[LLM] Ollama running — model: {self.model}"
            )

            installed_models: list[str] = []

            for model in model_response.models:
                model_name = getattr(
                    model,
                    "model",
                    "",
                )

                if model_name:
                    installed_models.append(model_name)

            if not any(
                self.model in name
                for name in installed_models
            ):
                print(
                    f"[LLM] Model {self.model} is not installed."
                )
                print(
                    f"[LLM] Pulling {self.model}. "
                    "This may take several minutes."
                )

                ollama.pull(self.model)

                print(
                    f"[LLM] {self.model} is ready."
                )

        except Exception as error:
            self.available = False

            print(
                f"[LLM] Ollama not running: {error}"
            )
            print(
                "[LLM] Open the Ollama application, "
                "then restart Flask."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Clinical report generation
    # ──────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        image_path: str,
        detections: list[dict[str, Any]],
        rag_context: list[dict[str, Any]],
        metadata: dict[str, Any],
        model_label: str,
    ) -> dict[str, Any]:
        """
        Generate a structured decision-support report.

        image_path is retained for compatibility and traceability.
        The current Qwen text model does not independently inspect the image.
        """

        _ = image_path

        detections_text = format_detections(
            detections
        )

        metadata_text = format_metadata(
            metadata
        )

        rag_text = format_rag(
            rag_context
        )

        tumor_detected = has_tumor_detection(
            detections
        )

        if tumor_detected:
            finding_instruction = (
                "Describe the detected model finding without presenting it "
                "as a confirmed diagnosis."
            )
        else:
            finding_instruction = (
                "Explain that no trained tumor class was detected, while "
                "making clear that this does not exclude other abnormalities."
            )

        prompt = f"""
You are a clinical decision-support reporting assistant.

The report is based on:

1. A YOLO26 brain MRI detector
2. Deterministic metadata and urgency rules
3. Medical passages retrieved by a FAISS semantic-search system

MODEL
{model_label}

DETECTIONS
{detections_text}

STRUCTURED METADATA
{metadata_text}

RETRIEVED MEDICAL EVIDENCE
{rag_text}

TASK
{finding_instruction}

Write a professional report using exactly these headings:

### AI Detection Summary
State what the model detected and the confidence value.

### Finding and Localization
Describe the detected class and bounding-box information when available.
Do not claim anatomical details that are not in the supplied data.

### Urgency Assessment
Use the urgency value and recommendation from STRUCTURED METADATA.
Do not change, increase, or reduce the supplied urgency level.

### Clinical Context
Summarize only information supported by the retrieved evidence.
Make clear that clinical correlation is required.

### Recommended Next Steps
Provide non-prescriptive next steps such as formal radiology review,
specialist consultation, complete MRI evaluation, or appropriate follow-up.

### Model Limitations
Mention image quality, acquisition differences, artifacts, dataset bias,
limited classes, and the possibility of false-positive or false-negative results.

### Disclaimer
State that this is an AI-generated decision-support output,
not a confirmed diagnosis or substitute for a radiologist or clinician.

MANDATORY SAFETY RULES

- Do not invent patient age, sex, symptoms, history, laboratory values,
  MRI sequence, lesion dimensions, or anatomical location.
- Do not assign a WHO grade.
- Do not determine a disease stage.
- Do not prescribe medicines.
- Do not provide drug dosages.
- Do not predict survival or life expectancy.
- Do not state that the patient definitely has cancer.
- Do not contradict the supplied metadata.
- Do not use information outside the supplied evidence.
- Keep the report concise, clear, and understandable.
""".strip()

        if not self.available:
            return {
                "report": self._demo_report(
                    detections=detections,
                    metadata=metadata,
                    model_label=model_label,
                ),
                "demo_mode": True,
            }

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Follow the supplied clinical safety rules. "
                            "Do not invent unsupported medical facts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                options={
                    "num_predict": 700,
                    "temperature": 0.1,
                    "top_p": 0.8,
                },
                stream=False,
            )

            report = response["message"]["content"].strip()

            if not report:
                report = (
                    "The LLM returned an empty report. "
                    "Please run the analysis again."
                )

            return {
                "report": report,
                "demo_mode": False,
            }

        except Exception as error:
            error_message = str(error)

            print(
                f"[LLM Analysis Error] {error_message}"
            )

            if (
                "connection" in error_message.lower()
                or "refused" in error_message.lower()
            ):
                report = (
                    "⚠️ Ollama is not currently reachable.\n\n"
                    "Open the Ollama application and restart Flask."
                )
            else:
                report = (
                    "⚠️ The clinical report could not be generated.\n\n"
                    f"Error: {error_message}"
                )

            return {
                "report": report,
                "demo_mode": True,
            }

    # ──────────────────────────────────────────────────────────────────────────
    # Follow-up chat
    # ──────────────────────────────────────────────────────────────────────────

    def chat(
        self,
        question: str,
        detections: list[dict[str, Any]],
        analysis_summary: str,
        model_label: str,
        image_path: Optional[str] = None,
    ) -> str:
        """Answer follow-up questions about the latest result."""

        _ = image_path

        city = detect_city(question)

        if city:
            return specialist_links(city)

        condition = format_detections(
            detections
        )

        prompt = f"""
You are a cautious medical decision-support assistant.

MODEL
{model_label}

AI DETECTIONS
{condition}

PREVIOUS REPORT
{analysis_summary[:4000]}

USER QUESTION
{question}

Answer clearly and briefly.

Safety rules:

- Do not provide a confirmed diagnosis.
- Do not prescribe medicine or give drug dosages.
- Do not assign a WHO grade or disease stage.
- Do not predict survival or life expectancy.
- Do not invent symptoms, medical history, age, sex, or scan details.
- Explain uncertainty where appropriate.
- For urgent symptoms such as seizures, sudden weakness,
  altered consciousness, severe sudden headache, or acute visual loss,
  advise immediate emergency medical evaluation.
- For clinical questions, recommend consultation with a qualified specialist.
""".strip()

        if not self.available:
            return (
                "⚠️ Ollama is not running.\n\n"
                "Open the Ollama application, then restart Flask."
            )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Provide cautious, non-prescriptive medical "
                            "decision-support information."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                options={
                    "num_predict": 350,
                    "temperature": 0.2,
                    "top_p": 0.8,
                },
                stream=False,
            )

            answer = response["message"]["content"].strip()

            if not answer:
                return (
                    "No answer was generated. "
                    "Please try asking the question again."
                )

            return answer

        except Exception as error:
            error_message = str(error)

            print(
                f"[LLM Chat Error] {error_message}"
            )

            if (
                "connection" in error_message.lower()
                or "refused" in error_message.lower()
            ):
                return (
                    "⚠️ Ollama is not running. "
                    "Open the Ollama application and try again."
                )

            return f"⚠️ Error: {error_message}"

    # ──────────────────────────────────────────────────────────────────────────
    # Fallback report
    # ──────────────────────────────────────────────────────────────────────────

    def _demo_report(
        self,
        detections: list[dict[str, Any]],
        metadata: dict[str, Any],
        model_label: str,
    ) -> str:
        """Return a deterministic fallback when Ollama is unavailable."""

        detection_text = format_detections(
            detections
        )

        finding = metadata.get(
            "finding",
            "Not available",
        )

        confidence = metadata.get(
            "confidence_percent",
            "Not available",
        )

        urgency = metadata.get(
            "urgency",
            "Not available",
        )

        recommendation = metadata.get(
            "recommendation",
            (
                "Formal review by a qualified radiologist "
                "or clinician is recommended."
            ),
        )

        return f"""
### AI Detection Summary

Model: **{model_label}**

{detection_text}

### Finding and Localization

Model finding: **{finding}**

Confidence: **{confidence}%**

### Urgency Assessment

Urgency: **{urgency}**

{recommendation}

### Clinical Context

The local LLM is unavailable, so a generated clinical-context summary
could not be produced.

### Recommended Next Steps

Obtain formal interpretation from a qualified radiologist and correlate
the result with the complete MRI study, symptoms, history, and clinical findings.

### Model Limitations

The detector may be affected by image quality, acquisition differences,
artifacts, dataset bias, and findings outside its trained classes.

### Disclaimer

This is an AI-generated decision-support result. It is not a confirmed
diagnosis and does not replace professional medical evaluation.

### LLM Status

Ollama is not reachable. Open the Ollama application and ensure that
`{MODEL_NAME}` is installed.
""".strip()