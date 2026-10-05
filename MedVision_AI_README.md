# MedVision AI

### Privacy-Conscious Brain MRI Decision-Support Prototype

MedVision AI is a locally deployed AI-based decision-support prototype for **brain MRI tumor detection and clinical-style reporting**.

The system combines **YOLO26 object detection**, structured metadata generation, rule-based urgency assessment, **FAISS-based Retrieval-Augmented Generation (RAG)**, and a locally executed **Qwen2.5 LLM** to transform MRI detection results into an interpretable clinical-style report.

> **Important:** MedVision AI is a research prototype and is not a clinically validated diagnostic system. It is intended for research and educational purposes only.

---

## 📌 Overview

MedVision AI follows a multi-stage AI pipeline:

```text
Brain MRI
    ↓
YOLO26 Tumor Detection
    ↓
Structured Detection Metadata
    ↓
Rule-Based Urgency Assessment
    ↓
SentenceTransformer Embeddings
    ↓
FAISS Retrieval
    ↓
Retrieved Medical Evidence
    ↓
Qwen2.5 via Ollama
    ↓
Clinical-Style Report
```

The **core AI inference pipeline runs locally**. Optional healthcare-resource navigation may use external web links or services.

---

## ✨ Key Features

- 🧠 Brain MRI tumor detection using YOLO26
- 📦 Bounding-box localization of detected lesions
- 📊 Confidence-based detection results
- 📝 Automatic structured metadata generation
- ⚠️ Rule-based urgency prioritization
- 🔎 Retrieval-Augmented Generation using FAISS
- 📚 Medical knowledge retrieval using SentenceTransformers
- 🤖 Local clinical-style report generation using Qwen2.5
- 🔒 Privacy-conscious local AI processing
- ⚡ GPU acceleration using CUDA when available
- 🏥 Optional healthcare-resource navigation

---

## 🧠 Supported Tumor Classes

| Class ID | Class |
|---:|---|
| 0 | Glioma |
| 1 | Meningioma |
| 2 | No Tumor |
| 3 | Pituitary |

---

## 🏗️ System Architecture

```text
                         ┌─────────────────┐
                         │    Brain MRI    │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │     YOLO26      │
                         │ Tumor Detector  │
                         └────────┬────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Structured Metadata │
                       │ • Class             │
                       │ • Confidence        │
                       │ • Bounding Box      │
                       │ • Detection Count   │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Urgency Assessment  │
                       │    Rule-Based       │
                       └──────────┬──────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ SentenceTransformer     │
                     │ all-MiniLM-L6-v2        │
                     └────────────┬────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │      FAISS      │
                         │ Evidence Search │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    Qwen2.5      │
                         │ 1.5B via Ollama │
                         └────────┬────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Clinical-Style      │
                       │ Report              │
                       └─────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Web Framework | Flask |
| Object Detection | YOLO26 / Ultralytics |
| Deep Learning | PyTorch |
| Embeddings | SentenceTransformer |
| Vector Search | FAISS |
| Large Language Model | Qwen2.5 1.5B |
| Local LLM Runtime | Ollama |
| Image Processing | OpenCV |
| Numerical Processing | NumPy |
| GPU Acceleration | NVIDIA CUDA |

---

## 📂 Project Structure

```text
MedVision_AI/
│
├── app.py
├── detector.py
├── metadata_engine.py
├── rag_engine.py
├── llm_agent.py
│
├── models/
│   └── brain_tumor_yolo26.pt
│
├── static/
├── templates/
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <YOUR_MEDVISION_REPOSITORY_URL>
cd MedVision_AI
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If GPU acceleration is required, install the appropriate PyTorch build for your NVIDIA GPU and CUDA environment.

---

# 🤖 Ollama Setup

MedVision AI uses **Qwen2.5 1.5B through Ollama** for local clinical-style report generation.

```bash
ollama pull qwen2.5:1.5b
```

Verify:

```bash
ollama list
```

Expected model:

```text
qwen2.5:1.5b
```

---

# ▶️ Running the Application

```bash
python app.py
```

Then open the local address displayed by Flask in your browser.

---

# 🔍 YOLO26 Detection

The deployed model is:

```text
models/brain_tumor_yolo26.pt
```

Configuration:

```text
Image Size:       640 × 640
Confidence:       0.25
IoU Threshold:    0.45
Device:           GPU when available
```

Ultralytics handles the detection pipeline and built-in Non-Maximum Suppression (NMS).

The application returns detected objects with confidence scores and bounding boxes.

---

# 📊 Detection Metadata

MedVision AI can generate structured metadata including:

- Detected pathology
- Confidence score
- Bounding box
- Detection count
- Brain MRI modality specification
- Urgency level
- Recommendation
- Processing/inference time

---

# ⚠️ Urgency Assessment

MedVision AI uses a **simple rule-based prioritization heuristic**.

| Detected Class | Prototype Urgency |
|---|---|
| Glioma | High |
| Meningioma | Moderate |
| Pituitary | Moderate |
| No Tumor | Routine |

This mechanism is **not clinical triage**.

Actual clinical urgency can depend on tumor grade, location, size, symptoms, mass effect, patient history, radiological findings, and other clinical information not determined by this prototype.

---

# 🔎 Retrieval-Augmented Generation (RAG)

MedVision AI uses RAG to provide relevant medical evidence to the reporting stage.

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Vector Database

```text
FAISS
IndexFlatIP
```

The knowledge base contains curated medical-information passages.

Runtime retrieval uses:

```text
Top-K = 4
```

The retrieval query is constructed from the detected pathology classes together with a fixed clinical-information prompt covering MRI interpretation, urgency, follow-up, and safety limitations.

---

# 📈 RAG Evaluation

A controlled evaluation used **39 medical-information queries**.

| Metric | Result |
|---|---:|
| Top-1 Accuracy | **92.31%** |
| Top-3 Recall | **100%** |
| Mean Reciprocal Rank (MRR) | **0.9573** |

These results represent a controlled retrieval evaluation and are not clinical diagnostic accuracy.

> Runtime retrieval uses Top-4 passages, while the independent evaluation reports Top-3 recall. These are separate evaluation settings.

---

# 🤖 Local LLM Reporting

MedVision AI uses:

```text
Qwen2.5 1.5B
        +
Ollama
```

The LLM receives structured detector results and retrieved medical evidence.

### Important

Qwen2.5 **does not directly analyze MRI pixels**. YOLO26 performs the image analysis.

```text
Brain MRI
    ↓
YOLO26
    ↓
Structured Findings
    ↓
RAG Evidence
    ↓
Qwen2.5
    ↓
Clinical-Style Report
```

Qwen2.5 is therefore used as a report-generation and explanation component rather than the primary image-diagnosis model.

---

# 🛡️ Reporting Safety Constraints

The reporting prompt is designed to prevent the LLM from:

- Inventing patient information
- Assigning unsupported WHO tumor grades
- Determining cancer stage
- Prescribing medication or dosage
- Predicting survival
- Claiming definite cancer diagnosis
- Contradicting detector metadata
- Generating unsupported clinical information

Generated reports should always be reviewed by an appropriately qualified medical professional.

---

# 📋 Reporting Evaluation

A controlled evaluation used **8 test cases**.

| Metric | Result |
|---|---:|
| Pathology Consistency | **100%** |
| Evidence Grounding | **91.67%** |
| Report Completeness | **100%** |
| Safety Compliance | **100%** |
| Overall Reporting Score | **97.92%** |

The 97.92% value is the overall reporting score under the defined evaluation protocol, not diagnostic accuracy.

---

# 🧪 Model Performance

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| YOLOv8 | 94.84% | **95.91%** | **96.80%** | **78.66%** |
| YOLO12 | 94.31% | 94.77% | 96.28% | 78.15% |
| YOLO26 | **95.83%** | 92.39% | 95.61% | 76.22% |

### Model Selection

YOLO26 was selected for deployment because it achieved the **highest precision** among the evaluated models.

YOLOv8 achieved higher recall and mAP, so YOLO26 should not be described as the overall best-performing model.

---

# 🎯 Lesion-Size Analysis

The associated model-development study evaluated three experimental lesion-size categories:

| Lesion Size | YOLOv8 Recall | YOLO12 Recall | YOLO26 Recall | Ground-Truth Lesions |
|---|---:|---:|---:|---:|
| Small (<1%) | 87.30% | 84.92% | 85.71% | 126 |
| Medium (1–3%) | 96.70% | 95.75% | 94.81% | 212 |
| Large (>3%) | 98.31% | 98.31% | 97.46% | 118 |

These are **study-defined experimental size groups**, not clinical tumor staging.

Smaller lesions were more difficult to detect than larger lesions in this analysis.

---

# 🔬 Very-Small Lesion Example

A representative very-small lesion occupied approximately:

```text
0.0797% of the image area
```

For this representative case:

| Model | Result |
|---|---|
| YOLOv8 | No class-correct overlapping detection |
| YOLO12 | No class-correct overlapping detection |
| YOLO26 | Confidence = 0.578, IoU = 0.684 |

This is a **single representative case** and should not be interpreted as proof that YOLO26 is generally superior for all very-small lesions.

---

# 🖥️ Hardware

Development and deployment used:

```text
GPU:       NVIDIA RTX 3050 Laptop GPU
PyTorch:   2.11.0+cu128
CUDA:      12.8
```

GPU acceleration depends on the local hardware and software configuration.

---

# 🔒 Privacy and Deployment

The core AI pipeline is designed for local execution:

```text
Brain MRI
    ↓
Local YOLO26
    ↓
Local Metadata Processing
    ↓
Local FAISS Retrieval
    ↓
Local Qwen2.5
    ↓
Clinical-Style Report
```

Local processing reduces the need to transmit MRI images to external AI APIs.

The complete application should **not** be described as strictly offline because optional healthcare-resource navigation may use external web links or services.

---

# 🏥 Healthcare Resource Navigation

The application can provide optional healthcare-resource navigation based on detected findings.

This functionality is intended as location-based navigation/search assistance, not as a ranking or recommendation of the "best" hospital.

---

# ⚠️ Limitations

1. The system has not undergone clinical validation.
2. The system is designed for brain MRI images.
3. The urgency mechanism is a simplified rule-based heuristic.
4. YOLO26 does not establish histopathological grade or molecular subtype.
5. Qwen2.5 does not directly interpret MRI pixels.
6. RAG evaluation used controlled queries rather than real patient cases.
7. A no-RAG baseline was not evaluated, so quantitative hallucination reduction due specifically to RAG has not been established.
8. The reporting evaluation uses a limited controlled test set.
9. External clinical validation is required before real-world medical use.
10. Performance may vary across datasets, scanners, acquisition protocols, and populations.

---

# 🚀 Future Work

- Larger and more diverse MRI datasets
- External held-out clinical evaluation
- Multi-center validation
- Improved small-lesion detection
- Better uncertainty estimation
- More clinically grounded urgency rules
- Larger medical knowledge bases
- Citation-aware report generation
- Clinician feedback integration
- Explainable AI visualizations
- Improved multimodal reporting
- Prospective clinical evaluation

---

# 🔗 Related Repository

The model training, evaluation, comparison, and lesion-size analysis are maintained separately in the **BrainTumorDetection** repository.

```text
BrainTumorDetection
        │
        │ Trained YOLO26 model
        ▼
MedVision AI
        │
        ├── Detection
        ├── Metadata
        ├── Urgency Assessment
        ├── RAG
        ├── Qwen2.5
        └── Clinical-Style Report
```

---

# 📚 Research Components

### 1. Computer Vision

YOLO26 performs object detection and localization on brain MRI images.

### 2. Structured Metadata

Detection outputs are converted into structured information containing pathology, confidence, bounding-box information, and processing metadata.

### 3. Rule-Based Prioritization

A rule-based mechanism assigns prototype urgency levels based on the detected class.

### 4. Retrieval-Augmented Generation

FAISS retrieves relevant medical-information passages using SentenceTransformer embeddings.

### 5. Local LLM

Qwen2.5 generates a clinical-style report using structured findings and retrieved evidence.

---

# 🔄 End-to-End Workflow

```text
                 USER
                  │
                  ▼
           Upload Brain MRI
                  │
                  ▼
          Image Preprocessing
                  │
                  ▼
              YOLO26
                  │
                  ▼
        Tumor Detection Results
                  │
                  ▼
       Structured Metadata Engine
                  │
                  ▼
       Rule-Based Urgency Level
                  │
                  ▼
        RAG Query Construction
                  │
                  ▼
      SentenceTransformer Encoder
                  │
                  ▼
             FAISS Search
                  │
                  ▼
        Retrieved Medical Evidence
                  │
                  ▼
        Qwen2.5 via Ollama
                  │
                  ▼
        Clinical-Style Report
                  │
                  ▼
              User Output
```

---

# 📌 Research Disclaimer

This project is intended for **research, educational, and prototype development purposes**.

MedVision AI is **not a medical device** and has not been clinically validated.

It must not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

Any medical interpretation or decision should be made by a qualified healthcare professional.

---

# 👩‍💻 Author

**Jasleen Kaur Khalsa**

B.Tech Computer Science (Artificial Intelligence)

---

# 📄 License

Add the appropriate license for this repository.

For example:

```text
MIT License
```

if the repository is intended to be released under the MIT License.
