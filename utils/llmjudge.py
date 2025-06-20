

import logging
import os

def judge_answer_with_api(question, target, answer):
    from openai import OpenAI

    client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY")
    )

    prompt = (
        "You will be given a question, a target answer (maybe a list of all possible answers), "
        "and a generated answer. Please judge whether the generated answer is correct. "
        "If it is correct, return 'True'. If it is incorrect, return 'False'.\n"
        f"Question: {question}\n"
        f"Target Answer: {target}\n"
        f"Generated Answer: {answer}\n"
        "Please return only 'True' or 'False', without any other text."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1
        )
    except Exception as e:
        logging.error("API call failed: %s", str(e))
        return 0  # Return default value

    try:
        result = response.choices[0].message.content.strip()
    except (AttributeError, IndexError) as e:
        logging.error("Error parsing API response: %s", str(e))
        return 0

    if result == "True":
        return 1
    elif result == "False":
        return 0
    else:
        logging.warning("Abnormal response format: %s", result)
        return 0


