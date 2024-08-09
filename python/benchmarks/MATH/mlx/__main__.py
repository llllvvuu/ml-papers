import json
import random
import os
import argparse
import mlx_lm
from tabulate import tabulate

from ...utils.extract_answer import extract_boxed_answer


def get_json_files(directory: str) -> list[str]:
    json_files: list[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def generate(model_path: str, prompt: str, verbose: bool = False) -> str:
    model, tokenizer = mlx_lm.load(model_path)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = mlx_lm.generate(
        model,
        tokenizer,
        prompt=text,
        verbose=verbose,
        max_tokens=1024,
    )
    return response


def main(model_path: str, dataset_path: str, sample_size: int) -> None:
    json_files = get_json_files(dataset_path)
    sample = random.sample(json_files, sample_size)

    n_correct = 0
    expected_answers: list[str] = []
    llm_answers: list[str] = []

    for file_path in sample:
        print(f"Running {file_path}.")
        with open(file_path, "r") as f:
            example = json.load(f)
        problem: str = example["problem"]
        print(f"Problem: {problem}")
        print(f"Level: {example['level']}")
        answer = extract_boxed_answer(example["solution"])
        print(f"Answer: {answer}")
        response = generate(model_path, problem, verbose=True)
        llm_answer = extract_boxed_answer(response)
        print(f"LLM's Answer: {llm_answer}")

        expected_answers.append(answer)
        llm_answers.append(llm_answer)

        if llm_answer == answer:
            n_correct += 1
            print("Correct!")
        else:
            print("Wrong...")

    print(f"Correct: {n_correct}/{sample_size}")

    table = list(zip(expected_answers, llm_answers))
    headers = ["Expected Answer", "LLM's Answer"]
    print("\nResults Table:")
    print(tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM on MATH problems")
    _ = parser.add_argument(
        "model_path", type=str, help="Example: _mlx/Qwen2-Math-1.5B-Instruct"
    )
    _ = parser.add_argument(
        "dataset_path", type=str, help="Example: _datasets/MATH/test"
    )
    _ = parser.add_argument(
        "sample_size", type=int, help="Number of problems to sample"
    )
    args = parser.parse_args()

    main(args.model_path, args.dataset_path, args.sample_size)
