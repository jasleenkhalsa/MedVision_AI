from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = Path("models/brain_tumor_yolo26.pt")

# Replace this with a real MRI image from your validation dataset.
IMAGE_PATH = Path(
    r"D:/BrainTumorDetection/dataset/processed/val/images/m2 (86).jpg"
)

OUTPUT_DIR = Path("results")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.resolve()}"
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            "Test image not found. Update IMAGE_PATH in test_yolo26.py.\n"
            f"Current value: {IMAGE_PATH}"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=640,
        conf=0.25,
        iou=0.45,
        device=0,
        project=str(OUTPUT_DIR),
        name="yolo26_test",
        exist_ok=True,
        save=True,
        verbose=True,
    )

    result = results[0]

    print("\nPrediction completed.")
    print(f"Classes: {result.names}")
    print(f"Number of boxes: {len(result.boxes)}")

    if result.boxes is None or len(result.boxes) == 0:
        print("No detection passed the confidence threshold.")
        return

    for number, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        coordinates = [
            round(value, 2)
            for value in box.xyxy[0].tolist()
        ]

        print(
            f"Detection {number}: "
            f"class={result.names[class_id]}, "
            f"confidence={confidence:.4f}, "
            f"bbox={coordinates}"
        )

    print(
        "\nAnnotated image saved under: "
        "results\\yolo26_test"
    )


if __name__ == "__main__":
    main()