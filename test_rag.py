from rag_engine import MedicalRAGEngine


def main() -> None:
    rag = MedicalRAGEngine()

    test_cases = [
        ["Glioma"],
        ["Meningioma"],
        ["Pituitary"],
        ["No Tumor"],
    ]

    for detected_classes in test_cases:
        print("\n" + "=" * 70)
        print("Detected class:", detected_classes[0])

        results = rag.get_context(
            detected_classes=detected_classes,
            model_key="brain_tumor",
            top_k=3,
        )

        for position, item in enumerate(results, start=1):
            print(
                f"\nResult {position}"
                f"\nCategory: {item['category']}"
                f"\nScore: {item['score']}"
                f"\nText: {item['text']}"
            )


if __name__ == "__main__":
    main()