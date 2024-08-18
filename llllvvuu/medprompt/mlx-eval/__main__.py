import json
import random
import os
import argparse
import mlx_lm
import torch
import torch.nn as nn
from tabulate import tabulate
from tqdm import tqdm
from mlx_embedding_models.embedding import EmbeddingModel

from ...benchmarks.utils.extract_answer import extract_boxed_answer


DEFAULT_SYSTEM_MESSAGE = "You are an elite competitive math teacher. You will be asked a series of math questions, all of which can be solved similarly. For exah problem, you should give the answer using \\boxed{} AFTER (NOT before) explaining how to solve it. For example, if the question is '2 + 2 = ?', your answer MUST end in '\\boxed{4}'. Think step-by-step and finish each problem in one response; do NOT ask the user if they want further explanation. Do NOT say anything like 'Let me know if you'd like me to walk through the calculations'. You MUST end your immediate response with your final answer in \\boxed{}, which MUST be nonempty and in reduced form."


def get_json_files(directory: str) -> list[str]:
    json_files: list[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def generate(
    model_path: str,
    instruct: bool,
    examples: list[tuple[str, str]],
    problem: str,
    system_message: str,
    verbose: bool = False,
) -> str:
    model, tokenizer = mlx_lm.load(model_path)
    if instruct:
        messages = [{"role": "system", "content": system_message}]
        for example_problem, example_solution in examples:
            messages.append({"role": "user", "content": example_problem})
            messages.append({"role": "assistant", "content": example_solution})
        messages.append({"role": "user", "content": problem})
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:  # noqa: E722
            messages[1]["content"] = (
                messages[0]["content"] + "\n\n" + messages[1]["content"]
            )
            messages = messages[1:]
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
        prompt = "In the following solution manual, each final answer is wrapped in \\boxed{}. The problems are all solved similarly."
        prompt += tokenizer.eos_token
        for example_problem, example_solution in examples:
            prompt += f"Problem: {example_problem}{tokenizer.eos_token}Solution: {example_solution}{tokenizer.eos_token}"
        prompt += f"Problem: {problem}{tokenizer.eos_token}Solution: "
        return mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            verbose=verbose,
            max_tokens=2048,
            top_p=0.95,  # TODO: `min_p` to be released soon
            temp=0.8,
        )


def embed_problem(problem: str, model: EmbeddingModel) -> torch.Tensor:
    return torch.tensor(model.encode([problem]).squeeze())


def find_nearest_neighbors(
    query_embedding: torch.Tensor, embeddings: dict[str, list[float]], k: int = 5
) -> list[str]:
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    similarities = torch.tensor(
        [cos(query_embedding, torch.tensor(emb)) for emb in embeddings.values()]
    )
    _, indices = torch.topk(similarities, k)
    return [list(embeddings.keys())[i] for i in indices.tolist()]


def get_examples(neighbor_files: list[str], dataset_path: str) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for file in neighbor_files:
        try:
            with open(os.path.join(dataset_path, file), "r") as f:
                example = json.load(f)
            examples.append((example["problem"], example["solution"]))
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"Error processing file {file}: {str(e)}")
    return examples


def process_problem(
    file_path: str,
    embeddings: dict,
    embedding_model: EmbeddingModel,
    train_dataset_path: str,
    model_path: str,
    instruct: bool,
    system_message: str,
    k_neighbors: int,
    verbose: bool,
) -> tuple[str, str, str]:
    try:
        with open(file_path, "r") as f:
            example = json.load(f)
        problem: str = example["problem"]
        answer = extract_boxed_answer(example["solution"])

        problem_embedding = embed_problem(problem, embedding_model)
        neighbor_files = find_nearest_neighbors(
            problem_embedding, embeddings, k_neighbors
        )
        examples = get_examples(neighbor_files, train_dataset_path)

        response = generate(
            model_path, instruct, examples, problem, system_message, verbose
        )
        llm_answer = extract_boxed_answer(response)

        return answer, llm_answer, example.get("level", "Unknown")
    except Exception as e:
        print(f"Error processing problem {file_path}: {str(e)}")
        return "Error", "Error", "Unknown"


def main(
    model_path: str,
    instruct: bool,
    embeddings_path: str,
    train_dataset_path: str,
    test_dataset_path: str,
    sample_size: int,
    system_message: str,
    k_neighbors: int,
    verbose: bool,
) -> None:
    json_files = get_json_files(test_dataset_path)
    sample = random.sample(json_files, sample_size)

    try:
        with open(embeddings_path, "r") as f:
            embeddings = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading embeddings: {str(e)}")
        return

    embedding_model = EmbeddingModel.from_registry("bge-small")

    results: list[tuple[str, str, str, bool]] = []
    n_correct = 0

    for i, file_path in enumerate(tqdm(sample)):
        answer, llm_answer, level = process_problem(
            file_path,
            embeddings,
            embedding_model,
            train_dataset_path,
            model_path,
            instruct,
            system_message,
            k_neighbors,
            verbose,
        )
        correct = llm_answer == answer
        if correct:
            print(f"🎉 Correct! Level: {level}, Answer: {answer}")
            n_correct += 1
        else:
            print(
                f"❌ Incorrect! Level: {level}, Answer: {answer}, LLM's Answer: {llm_answer}"
            )
        results.append((answer, llm_answer, level, correct))
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
    _ = parser.add_argument("embedding_path", type=str, help="Example: embeddings.json")
    _ = parser.add_argument(
        "train_dataset_path", type=str, help="Example: _datasets/MATH/train"
    )
    _ = parser.add_argument(
        "test_dataset_path", type=str, help="Example: _datasets/MATH/test"
    )
    _ = parser.add_argument(
        "sample_size", type=int, help="Number of problems to sample"
    )
    _ = parser.add_argument("--instruct", action="store_true")
    _ = parser.add_argument(
        "--system_message",
        type=str,
        default=DEFAULT_SYSTEM_MESSAGE,
    )
    _ = parser.add_argument(
        "--k_neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbors to consider",
    )
    _ = parser.add_argument(
        "--quiet", action="store_true", help="Disable verbose output"
    )
    args = parser.parse_args()

    main(
        args.model_path,
        args.instruct,
        args.embedding_path,
        args.train_dataset_path,
        args.test_dataset_path,
        args.sample_size,
        args.system_message,
        args.k_neighbors,
        not args.quiet,
    )
