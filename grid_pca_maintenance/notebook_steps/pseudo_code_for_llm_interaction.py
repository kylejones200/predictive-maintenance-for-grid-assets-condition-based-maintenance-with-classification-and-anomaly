"""Notebook steps (auto-split)."""


def pseudo_code_for_llm_interaction() -> None:
    prompt = "\nYou are looking at a time series forecasting model for sensor_9 from a jet engine.\nThe model starts to deviate significantly from actual values after cycle 150.\nWhat might cause this, and what could you try to improve it?\n"
    response = openai.ChatCompletion.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )
    print(response["choices"][0]["message"]["content"])
