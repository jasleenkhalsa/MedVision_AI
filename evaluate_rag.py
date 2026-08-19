from rag_engine import MedicalRAGEngine


# ---------------------------------------------------------
# Test dataset
# ---------------------------------------------------------

TEST_CASES = [

    # ==================== GLIOMA ====================

    {
        "query": "What is a glioma?",
        "expected": "glioma_definition"
    },
    {
        "query": "Where do gliomas originate?",
        "expected": "glioma_definition"
    },
    {
        "query": "What MRI features are considered when assessing glioma?",
        "expected": "glioma_definition"
    },

    {
        "query": "How is a suspected glioma evaluated and managed?",
        "expected": "glioma_management"
    },
    {
        "query": "What further evaluation may be required for glioma?",
        "expected": "glioma_management"
    },
    {
        "query": "Does a glioma require specialist assessment?",
        "expected": "glioma_management"
    },

    {
        "query": "When does a suspected glioma require urgent medical attention?",
        "expected": "glioma_urgency"
    },
    {
        "query": "Are seizures with a glioma an urgent finding?",
        "expected": "glioma_urgency"
    },
    {
        "query": "What symptoms make a glioma case urgent?",
        "expected": "glioma_urgency"
    },


    # ==================== MENINGIOMA ====================

    {
        "query": "What is a meningioma?",
        "expected": "meningioma_definition"
    },
    {
        "query": "Where do meningiomas arise?",
        "expected": "meningioma_definition"
    },
    {
        "query": "What determines the clinical significance of a meningioma?",
        "expected": "meningioma_definition"
    },

    {
        "query": "How is a suspected meningioma managed?",
        "expected": "meningioma_management"
    },
    {
        "query": "What treatment options may be considered for meningioma?",
        "expected": "meningioma_management"
    },
    {
        "query": "Can meningioma management include imaging surveillance?",
        "expected": "meningioma_management"
    },

    {
        "query": "When does a meningioma require urgent assessment?",
        "expected": "meningioma_urgency"
    },
    {
        "query": "Are seizures or visual disturbance concerning in meningioma?",
        "expected": "meningioma_urgency"
    },
    {
        "query": "What symptoms make meningioma urgent?",
        "expected": "meningioma_urgency"
    },


    # ==================== PITUITARY ====================

    {
        "query": "What is a pituitary tumor?",
        "expected": "pituitary_definition"
    },
    {
        "query": "What structures can a pituitary tumor affect?",
        "expected": "pituitary_definition"
    },
    {
        "query": "What investigations are commonly used for pituitary tumors?",
        "expected": "pituitary_definition"
    },

    {
        "query": "How is a suspected pituitary lesion managed?",
        "expected": "pituitary_management"
    },
    {
        "query": "What treatment approaches may be used for pituitary lesions?",
        "expected": "pituitary_management"
    },
    {
        "query": "Should pituitary lesions be correlated with endocrine symptoms?",
        "expected": "pituitary_management"
    },

    {
        "query": "When is a pituitary tumor an emergency?",
        "expected": "pituitary_urgency"
    },
    {
        "query": "Does sudden visual loss with a pituitary lesion require emergency evaluation?",
        "expected": "pituitary_urgency"
    },
    {
        "query": "What symptoms suggest pituitary apoplexy requiring urgent care?",
        "expected": "pituitary_urgency"
    },


    # ==================== NO TUMOR ====================

    {
        "query": "What does a no tumor prediction mean?",
        "expected": "no_tumor_interpretation"
    },
    {
        "query": "Does a no tumor prediction exclude all abnormalities?",
        "expected": "no_tumor_interpretation"
    },
    {
        "query": "Can the model miss small lesions even when it predicts no tumor?",
        "expected": "no_tumor_interpretation"
    },

    {
        "query": "What should a patient do if symptoms continue despite no tumor detection?",
        "expected": "no_tumor_followup"
    },
    {
        "query": "Should automated no tumor detection replace radiological interpretation?",
        "expected": "no_tumor_followup"
    },
    {
        "query": "Is clinical follow-up necessary when the model reports no tumor?",
        "expected": "no_tumor_followup"
    },


    # ==================== GENERAL / SAFETY ====================

    {
        "query": "What factors can affect automated MRI analysis?",
        "expected": "mri_limitations"
    },
    {
        "query": "Can patient motion or image artifacts affect MRI model predictions?",
        "expected": "mri_limitations"
    },
    {
        "query": "How can image quality influence automated MRI analysis?",
        "expected": "mri_limitations"
    },

    {
        "query": "Can an AI system replace a radiologist?",
        "expected": "clinical_safety"
    },
    {
        "query": "What information should be considered before making a clinical conclusion?",
        "expected": "clinical_safety"
    },
    {
        "query": "Should AI independently prescribe medication?",
        "expected": "clinical_safety"
    },
]


def main():

    print("=" * 75)
    print("MedVision AI - RAG Retrieval Evaluation")
    print("=" * 75)

    rag = MedicalRAGEngine()

    total = len(TEST_CASES)

    top1_correct = 0
    top3_correct = 0
    reciprocal_rank_sum = 0.0

    print(f"\nTotal evaluation queries: {total}\n")

    for number, test in enumerate(TEST_CASES, start=1):

        query = test["query"]
        expected = test["expected"]

        results = rag.search(
            query=query,
            top_k=3,
            minimum_score=0.0
        )

        retrieved_ids = [
            result["id"]
            for result in results
        ]

        # -------------------------
        # Top-1 accuracy
        # -------------------------

        if retrieved_ids and retrieved_ids[0] == expected:
            top1_correct += 1
            top1_status = "PASS"
        else:
            top1_status = "FAIL"

        # -------------------------
        # Top-3 recall
        # -------------------------

        if expected in retrieved_ids:
            top3_correct += 1
            rank = retrieved_ids.index(expected) + 1
            reciprocal_rank_sum += 1 / rank
        else:
            rank = None

        print("-" * 75)

        print(f"Test {number}")
        print(f"Query       : {query}")
        print(f"Expected    : {expected}")
        print(f"Retrieved   : {retrieved_ids}")

        if rank:
            print(f"Expected rank: {rank}")
        else:
            print("Expected rank: NOT RETRIEVED")

        print(f"Top-1       : {top1_status}")

    # ---------------------------------------------------------
    # Final metrics
    # ---------------------------------------------------------

    top1_accuracy = (top1_correct / total) * 100
    top3_recall = (top3_correct / total) * 100
    mrr = reciprocal_rank_sum / total

    print("\n")
    print("=" * 75)
    print("FINAL RAG EVALUATION RESULTS")
    print("=" * 75)

    print(f"Total Queries       : {total}")
    print(f"Top-1 Correct       : {top1_correct}/{total}")
    print(f"Top-3 Correct       : {top3_correct}/{total}")

    print()
    print(f"Top-1 Accuracy      : {top1_accuracy:.2f}%")
    print(f"Top-3 Recall        : {top3_recall:.2f}%")
    print(f"Mean Reciprocal Rank: {mrr:.4f}")

    print("=" * 75)


if __name__ == "__main__":
    main()