import ollama


def main() -> None:
    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "user",
                "content": "Explain glioma in two simple sentences.",
            }
        ],
        options={
            "temperature": 0.2,
            "num_predict": 100,
        },
        stream=False,
    )

    print("Full response:")
    print(response)

    print("\nMessage content:")
    print(repr(response["message"]["content"]))


if __name__ == "__main__":
    main()