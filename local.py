
import pandas as pd
import ollama
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

PROMPT_FILE = DATASET_DIR / "prompts.csv"
RESPONSE_FILE = DATASET_DIR / "llm_responses.csv"

prompts = pd.read_csv(
    PROMPT_FILE,
    sep="\t",
    encoding="utf-8-sig"
)

prompts.columns = prompts.columns.str.strip().str.upper()

print("CSV file loaded successfully!")

responses = []

for _, row in prompts.iterrows():
    prompt_id = row["PROMPT_ID"]
    domain = row["DOMAIN"]
    prompt = str(row["PROMPT"]).strip()

    print(f"\nPrompt ID: {prompt_id}")
    print(f"Domain: {domain}")
    print(f"Prompt: {prompt}")

    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={"num_predict": 150}
        )

        answer = response["message"]["content"]
        print("Response:", answer)

    except Exception as error:
        answer = f"ERROR: {error}"
        print(answer)

    responses.append({
        "PROMPT_ID": prompt_id,
        "DOMAIN": domain,
        "PROMPT": prompt,
        "RESPONSE": answer
    })

result = pd.DataFrame(responses)

result.to_csv(
    RESPONSE_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nAll prompts processed!")
print("Responses saved to:", RESPONSE_FILE)
HISTORY_FILE = DATASET_DIR / "conversation_history.txt"
with open(HISTORY_FILE, "a", encoding="utf-8") as file:
    file.write(f"PROMPT: {prompt}\n")
    file.write(f"RESPONSE: {answer}\n")
    file.write("-" * 50 + "\n")
    result.to_csv(RESPONSE_FILE, index=False, encoding="utf-8-sig")

COMPARISON_FILE = DATASET_DIR / "response_comparison.csv"
result.to_csv(COMPARISON_FILE, index=False, encoding="utf-8-sig")

print("\nAll prompts processed!")
print("Responses saved to:", RESPONSE_FILE)
print("Comparison saved to:", COMPARISON_FILE)