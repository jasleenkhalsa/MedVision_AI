"""
rag_engine.py — Medical knowledge retrieval for detected tumor classes
Provides clinical context injected alongside detection results
"""

import math
import re
from typing import List, Dict

KNOWLEDGE_BASE = [
    {
        "id": "glioma_001",
        "tags": ["glioma", "brain_tumor", "mri", "glioblastoma"],
        "title": "Glioma — Overview & Imaging Features",
        "content": (
            "Gliomas are primary brain tumors arising from glial cells (astrocytes, oligodendrocytes, ependymal cells). "
            "They are graded WHO I–IV. Glioblastoma (WHO IV) is the most aggressive. On MRI: irregular ring-enhancing "
            "mass with central necrosis and surrounding vasogenic edema (T2/FLAIR hyperintense). "
            "Corpus callosum involvement ('butterfly glioma') is characteristic. Median survival GBM: 14–16 months. "
            "Treatment: maximal safe resection + temozolomide + radiotherapy (Stupp protocol)."
        ),
        "source": "WHO Classification of Tumors of the CNS 2021",
    },
    {
        "id": "glioma_002",
        "tags": ["glioma", "grading", "diffuse"],
        "title": "Glioma WHO Grading on Imaging",
        "content": (
            "Low-grade gliomas (WHO I–II): non-enhancing, T2 hyperintense, no necrosis, minimal mass effect. "
            "High-grade (WHO III–IV): irregular enhancement, necrosis, significant edema, crosses midline. "
            "IDH mutation status (IDH-mutant vs IDH-wildtype) is now central to classification. "
            "MR spectroscopy shows elevated choline, reduced NAA, lactate peak in high-grade. "
            "Perfusion MRI (rCBV) correlates with tumor grade."
        ),
        "source": "Neuroradiology Essentials 2024",
    },
    {
        "id": "meningioma_001",
        "tags": ["meningioma", "brain_tumor", "mri", "dural"],
        "title": "Meningioma — Imaging & Clinical Features",
        "content": (
            "Meningiomas arise from arachnoidal cap cells of the meninges. Most common extra-axial intracranial tumor. "
            "MRI: isointense on T1 and T2 to cortex, homogeneous intense enhancement with Gd, classic dural tail sign. "
            "Typically well-circumscribed, broad-based dural attachment. May cause adjacent hyperostosis. "
            "WHO grade I (benign, 80%) vs grade II (atypical) vs grade III (anaplastic). "
            "Treatment: observation for small/asymptomatic; surgery ± radiosurgery for symptomatic."
        ),
        "source": "Neurooncology Reference Atlas 2023",
    },
    {
        "id": "meningioma_002",
        "tags": ["meningioma", "locations", "differential"],
        "title": "Meningioma Common Locations & Differentials",
        "content": (
            "Common sites: parasagittal/falcine (most common), convexity, sphenoid wing, olfactory groove, "
            "cerebellopontine angle, tentorial, intraventricular. "
            "Differentials for extra-axial enhancing mass: dural metastasis, lymphoma, solitary fibrous tumor. "
            "Calcification on CT in up to 25%. Peritumoral edema more common in high-grade. "
            "Angiography may show 'sunburst' vascular pattern from external carotid supply."
        ),
        "source": "Neurosurgery Imaging Atlas",
    },
    {
        "id": "no_tumor_001",
        "tags": ["no_tumor", "normal", "brain", "mri"],
        "title": "Normal Brain MRI Anatomy",
        "content": (
            "Normal brain MRI: symmetric gray and white matter differentiation, no focal signal abnormalities, "
            "no midline shift, normal ventricular size and morphology, intact cortical sulcal pattern. "
            "T1: gray matter isointense, white matter hyperintense. T2/FLAIR: CSF bright, white matter dark. "
            "Normal enhancing structures: choroid plexus, pituitary, dural venous sinuses. "
            "Age-related changes: periventricular white matter T2 hyperintensities and mild cortical atrophy are normal variants."
        ),
        "source": "Neuroradiology Atlas 2024",
    },
    {
        "id": "pituitary_001",
        "tags": ["pituitary", "brain_tumor", "mri", "sellar"],
        "title": "Pituitary Tumor — Adenoma Imaging",
        "content": (
            "Pituitary adenomas are benign tumors of the anterior pituitary. Microadenomas (<10 mm): focal T1 "
            "hypointense lesion with delayed enhancement on dynamic Gd-MRI. Macroadenomas (>10 mm): may extend "
            "suprasellarly compressing optic chiasm (bitemporal hemianopia), invade cavernous sinuses. "
            "Functional adenomas: prolactinoma (most common), GH-secreting (acromegaly), ACTH-secreting (Cushing's). "
            "Non-functioning adenomas treated surgically (transsphenoidal). Prolactinomas: dopamine agonists first-line."
        ),
        "source": "Endocrine Radiology Handbook 2023",
    },
    {
        "id": "pituitary_002",
        "tags": ["pituitary", "sellar", "differential"],
        "title": "Sellar Region Differentials",
        "content": (
            "Sellar/parasellar mass differentials: pituitary adenoma (most common), craniopharyngioma "
            "(Rathke cleft cyst, calcification, children/young adults), meningioma (dural tail), "
            "aneurysm (ICA), germ cell tumor, hypothalamic glioma, metastasis. "
            "Rathke cleft cyst: T1 hyperintense, no enhancement, intracystic nodule. "
            "Craniopharyngioma: adamantinomatous (calcification + cysts, pediatric) vs papillary (adults, solid)."
        ),
        "source": "Neuroradiology Essentials 2024",
    },
    {
        "id": "yolo_brain_001",
        "tags": ["yolov8", "detection", "brain_tumor", "ai"],
        "title": "YOLOv8 Brain Tumor Detection — Model Notes",
        "content": (
            "YOLOv8 (You Only Look Once v8) is a real-time object detection architecture by Ultralytics. "
            "For brain tumor detection, it classifies MRI slices into: Glioma, Meningioma, No Tumor, Pituitary. "
            "Model inputs: 350×350px resized MRI images. Output: bounding boxes + class + confidence score. "
            "Confidence threshold (default 0.25) filters low-confidence predictions. "
            "Performance metrics: mAP@50, precision, recall per class evaluated on held-out test set. "
            "Always validate AI predictions against radiologist review."
        ),
        "source": "Ultralytics YOLOv8 Documentation",
    },
]


class TFIDFIndex:
    def __init__(self, docs):
        self.docs = docs
        self.idf: dict = {}
        self.tfidf: list = []
        self._build()

    def _tok(self, text: str) -> List[str]:
        sw = {"a","an","the","is","in","on","of","to","and","or","for","with","as","at",
              "by","this","that","are","was","it","be","from","has","have","can","may","also"}
        return [w for w in re.findall(r'\b[a-z][a-z0-9]*\b', text.lower())
                if w not in sw and len(w) > 2]

    def _build(self):
        N = len(self.docs)
        df: dict = {}
        corpus = []
        for d in self.docs:
            tokens = self._tok(f"{d['title']} {d['content']} {' '.join(d.get('tags', []))}")
            corpus.append(tokens)
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((N + 1) / (v + 1)) + 1 for t, v in df.items()}
        for tokens in corpus:
            tf: dict = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            L = len(tokens) or 1
            self.tfidf.append({t: (c / L) * self.idf.get(t, 1) for t, c in tf.items()})

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        q = self._tok(query)
        scores = {}
        for i, vec in enumerate(self.tfidf):
            s = sum(vec.get(t, 0) * self.idf.get(t, 1) for t in q)
            if s > 0:
                scores[i] = s
        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [dict(self.docs[i], score=round(scores[i], 4)) for i in ranked]


class MedicalRAGEngine:
    def __init__(self):
        self.index = TFIDFIndex(KNOWLEDGE_BASE)

    def get_context(self, detected_classes: List[str], model_key: str) -> List[Dict]:
        """Return relevant knowledge cards for the detected classes."""
        query = " ".join(detected_classes) + f" {model_key} brain tumor mri"
        results = self.index.search(query, top_k=4)
        return [
            {
                "title": r["title"],
                "content": r["content"],
                "source": r["source"],
                "relevance": r["score"],
            }
            for r in results
        ]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        return self.index.search(query, top_k=top_k)
