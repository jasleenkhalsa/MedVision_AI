from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = Path("models/brain_tumor_yolo26.pt")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH.resolve()}"
        )

    model = YOLO(str(MODEL_PATH))

    print("\nModel loaded successfully.")
    print(f"Model path: {MODEL_PATH.resolve()}")
    print(f"Model classes: {model.names}")
    print(f"Number of classes: {len(model.names)}")


if __name__ == "__main__":
    main()