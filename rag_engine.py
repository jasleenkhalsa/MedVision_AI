from __future__ import annotations

from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class MedicalRAGEngine:
    """
    Embedding-based medical retrieval engine.

    Pipeline:
    Medical passages
        -> SentenceTransformer embeddings
        -> FAISS vector index
        -> semantic retrieval
    """

    def __init__(self) -> None:
        print("[RAG] Loading embedding model...")

        self.embedding_model_name = (
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name
        )

        self.documents = self._build_knowledge_base()

        self.document_texts = [
            document["text"] for document in self.documents
        ]

        self.index = self._build_index()

        print(
            f"[RAG] FAISS index created with "
            f"{self.index.ntotal} medical passages"
        )

    def _build_knowledge_base(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "glioma_definition",
                "category": "Glioma",
                "source": "Clinical knowledge base",
                "text": (
                    "Gliomas are tumors that originate from glial cells "
                    "within the central nervous system. They include several "
                    "subtypes with different biological behaviours. MRI "
                    "assessment commonly considers lesion location, signal "
                    "characteristics, enhancement, surrounding edema, mass "
                    "effect and infiltration."
                ),
            },
            {
                "id": "glioma_management",
                "category": "Glioma",
                "source": "Clinical knowledge base",
                "text": (
                    "A suspected glioma requires specialist interpretation "
                    "and correlation with clinical history and complete MRI "
                    "sequences. Additional evaluation may include contrast "
                    "enhanced MRI, advanced imaging and histopathological "
                    "assessment. Treatment decisions must be made by the "
                    "appropriate multidisciplinary clinical team."
                ),
            },
            {
                "id": "glioma_urgency",
                "category": "Glioma",
                "source": "Clinical knowledge base",
                "text": (
                    "Urgency for a suspected glioma depends on neurological "
                    "symptoms, lesion size, edema, hydrocephalus, hemorrhage "
                    "and mass effect. New neurological deficits, seizures, "
                    "reduced consciousness or rapid clinical deterioration "
                    "require urgent medical assessment."
                ),
            },
            {
                "id": "meningioma_definition",
                "category": "Meningioma",
                "source": "Clinical knowledge base",
                "text": (
                    "Meningiomas commonly arise from the meninges surrounding "
                    "the brain and spinal cord. Many are slow growing, but "
                    "their clinical significance depends on size, location, "
                    "mass effect, surrounding edema and associated symptoms."
                ),
            },
            {
                "id": "meningioma_management",
                "category": "Meningioma",
                "source": "Clinical knowledge base",
                "text": (
                    "Management of a suspected meningioma may include imaging "
                    "surveillance, specialist consultation, surgery or other "
                    "treatment depending on lesion characteristics and the "
                    "patient's condition. Automated detection alone cannot "
                    "determine the treatment plan."
                ),
            },
            {
                "id": "meningioma_urgency",
                "category": "Meningioma",
                "source": "Clinical knowledge base",
                "text": (
                    "A meningioma may require urgent assessment when there is "
                    "significant mass effect, worsening neurological symptoms, "
                    "seizures, visual disturbance or altered consciousness. "
                    "Otherwise, specialist clinical review is generally "
                    "required to determine appropriate follow-up."
                ),
            },
            {
                "id": "pituitary_definition",
                "category": "Pituitary",
                "source": "Clinical knowledge base",
                "text": (
                    "Pituitary tumors arise in or near the pituitary gland. "
                    "They may affect hormone production or compress adjacent "
                    "structures such as the optic pathways. Evaluation often "
                    "includes dedicated pituitary MRI, endocrine testing and "
                    "visual assessment."
                ),
            },
            {
                "id": "pituitary_management",
                "category": "Pituitary",
                "source": "Clinical knowledge base",
                "text": (
                    "A suspected pituitary lesion should be clinically "
                    "correlated with endocrine symptoms, laboratory findings "
                    "and visual function. Management may involve observation, "
                    "medication, surgery or other specialist-directed care."
                ),
            },
            {
                "id": "pituitary_urgency",
                "category": "Pituitary",
                "source": "Clinical knowledge base",
                "text": (
                    "Sudden severe headache, acute visual loss, altered "
                    "consciousness or signs of pituitary apoplexy require "
                    "emergency medical evaluation. Stable suspected pituitary "
                    "lesions still require specialist endocrine and imaging "
                    "assessment."
                ),
            },
            {
                "id": "no_tumor_interpretation",
                "category": "No Tumor",
                "source": "Clinical knowledge base",
                "text": (
                    "A no-tumor model prediction means the detector did not "
                    "identify one of its trained tumor classes in the supplied "
                    "image. It does not exclude other abnormalities, small "
                    "lesions, artifacts or findings outside the model's "
                    "training scope."
                ),
            },
            {
                "id": "no_tumor_followup",
                "category": "No Tumor",
                "source": "Clinical knowledge base",
                "text": (
                    "The absence of an automated tumor detection should not "
                    "replace formal radiological interpretation. Persistent "
                    "or concerning symptoms should be evaluated by a qualified "
                    "healthcare professional even when the model reports no "
                    "tumor."
                ),
            },
            {
                "id": "mri_limitations",
                "category": "General",
                "source": "Clinical knowledge base",
                "text": (
                    "Automated MRI analysis may be affected by image quality, "
                    "scan sequence, patient motion, acquisition differences, "
                    "artifacts and dataset bias. Predictions must be treated "
                    "as decision-support information rather than a confirmed "
                    "diagnosis."
                ),
            },
            {
                "id": "clinical_safety",
                "category": "General",
                "source": "Clinical knowledge base",
                "text": (
                    "Clinical conclusions should combine imaging findings, "
                    "patient history, symptoms, examination, laboratory data "
                    "and specialist interpretation. An artificial intelligence "
                    "system must not independently prescribe medication or "
                    "replace a radiologist."
                ),
            },
        ]

    def _build_index(self) -> faiss.IndexFlatIP:
        embeddings = self.embedding_model.encode(
            self.document_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        return index

    def search(
        self,
        query: str,
        top_k: int = 4,
        minimum_score: float = 0.20,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        number_of_results = min(top_k, len(self.documents))

        scores, indices = self.index.search(
            query_embedding,
            number_of_results,
        )

        retrieved_documents: list[dict[str, Any]] = []

        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0:
                continue

            score_value = float(score)

            if score_value < minimum_score:
                continue

            document = self.documents[int(index_position)]

            retrieved_documents.append({
                "id": document["id"],
                "category": document["category"],
                "source": document["source"],
                "text": document["text"],
                "score": round(score_value, 4),
            })

        return retrieved_documents

    def get_context(
        self,
        detected_classes: list[str],
        model_key: str = "brain_tumor",
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Preserve the interface already used by app.py.
        """

        if not detected_classes:
            query = (
                "Brain MRI analysis with no detection. Explain model "
                "limitations, safe interpretation and appropriate follow-up."
            )
        else:
            class_text = ", ".join(detected_classes)

            query = (
                f"Brain MRI model findings: {class_text}. "
                "Retrieve clinically relevant information about the finding, "
                "MRI interpretation, urgency, follow-up and safety limitations."
            )

        return self.search(
            query=query,
            top_k=top_k,
        )