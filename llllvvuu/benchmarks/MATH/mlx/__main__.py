import json
import random
import os
import argparse
import mlx_lm
from tabulate import tabulate
from tqdm import tqdm

from ...utils.extract_answer import extract_boxed_answer


def get_json_files(directory: str) -> list[str]:
    json_files: list[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def generate(
    model_path: str, instruct: bool, problem: str, verbose: bool = False
) -> str:
    model, tokenizer = mlx_lm.load(model_path)
    if instruct:
        messages = [
            {
                "role": "system",
                "content": "You are an elite competitive math teacher. You will be asked a series of math questions, all of which can be solved similarly. For exah problem, you should give the answer using \\boxed{} AFTER (NOT before) explaining how to solve it. For example, if the question is '2 + 2 = ?', your answer MUST end in '\\boxed{4}'. Think step-by-step and finish each problem in one response; do NOT ask the user if they want further explanation. Do NOT say anything like 'Let me know if you'd like me to walk through the calculations'. You MUST end your immediate response with your final answer in \\boxed{}, which MUST be nonempty and in reduced form.",
            },
            {
                "role": "user",
                "content": problem,
            },
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:  # noqa: E722
            messages = [
                {
                    "role": "user",
                    "content": "I will ask a math problem, and I want you to give the answer using \\boxed{} AFTER (NOT before) explaining how to solve it. For example, if the problem is '2 + 2 = ?' then your response MUST end in '\\boxed{4}'. Think step-by-step and finish the problem in one response; do NOT ask me if I want any further information. Do NOT say anything like 'Let me know if you'd like me to walk through the calculations'. You MUST end your immediate response with your final answer in \\boxed{}, which MUST be nonempty and in reduced form.\n\nProblem: "
                    + problem,
                }
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return mlx_lm.generate(
            model,
            tokenizer,
            prompt=text,
            verbose=verbose,
            max_tokens=2048,
            top_p=0.95,  # TODO: `min_p` to be released soon
            temp=0.8,
        )
    else:
        # Conditions the models towards the desired output format.
        # Unfortunately, this doesn't work 100% of the time;
        # some models occasionally fail to generate an EOS token after answering,
        # and sometimes even fail to answer altogether.
        # TODO: Investigate:
        # - Stopping generation when a closed $\boxed{}$ is detected.
        # - Constraining generation.
        # XXX: Qwen2 has no trouble zero-shotting the correct answer format,
        # and these examples cause it to generate shorter solutions,
        # which actually seems to worsen performance.
        FORMATTING_EXAMPLES = [
            (
                "What is the sum of 2 and 3?",
                "The sum of 2 and 3 is $2 + 3 = \\boxed{5}$.",
            ),
            (
                "Solve x + 1 = 2 for x.",
                "Subtract 1 from both sides to get $x = \\boxed{1}$.",
            ),
            (
                "One of the acute angles in a right triangle measures 30 degrees. What is the measure of the other acute angle?",
                "The angles of a triangle sum to 180 degrees. Therefore, the other acute angle measures $180 - 90 - 30 = \\boxed{60}$ degrees.",
            ),
        ]
        prompt = "In the following solution manual, each final answer is wrapped in \\boxed{}."
        prompt += tokenizer.eos_token
        for example_problem, example_solution in FORMATTING_EXAMPLES:
            prompt += f"Problem: {example_problem}{tokenizer.eos_token}Solution: {example_solution}{tokenizer.eos_token}"
        prompt += f"Problem: {problem}{tokenizer.eos_token}Solution: "
        return mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            verbose=verbose,
            max_tokens=1024,
            top_p=0.95,  # TODO: `min_p` to be released soon
            temp=0.8,
        )


def main(model_path: str, instruct: bool, dataset_path: str, sample_size: int) -> None:
    json_files = get_json_files(dataset_path)
    sample = random.sample(json_files, sample_size)

    n_correct = 0
    results: list[tuple[str, str, str, bool]] = []

    for i, file_path in enumerate(tqdm(sample)):
        print(f"Running {file_path}.")
        with open(file_path, "r") as f:
            example = json.load(f)
        problem: str = example["problem"]
        print(f"Problem: {problem}")
        print(f"Level: {example['level']}")
        answer = extract_boxed_answer(example["solution"])
        print(f"Answer: {answer}")
        response = generate(model_path, instruct, problem, verbose=True)
        llm_answer = extract_boxed_answer(response)
        print(f"LLM's Answer: {llm_answer}")
        correct = llm_answer == answer

        if correct:
            print(f"🎉 Correct! Level: {example['level']}, Answer: {answer}")
            n_correct += 1
        else:
            print(
                f"❌ Incorrect! Level: {example['level']}, Answer: {answer}, LLM's Answer: {llm_answer}"
            )

        results.append((answer, llm_answer, example["level"], correct))
        print("=== RUNNING TALLY ===")
        print(f"Correct: {n_correct}/{i + 1}")
        headers = ["Expected Answer", "LLM's Answer", "Problem Level", "Correct?"]
        print("\nResults Table:")
        print(tabulate(results, headers=headers, tablefmt="grid"))


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
    _ = parser.add_argument("--instruct", action="store_true")
    args = parser.parse_args()

    main(args.model_path, args.instruct, args.dataset_path, args.sample_size)
