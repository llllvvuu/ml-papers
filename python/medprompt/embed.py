import argparse
import json
import os
from mlx_embedding_models.embedding import EmbeddingModel
from tqdm import tqdm

model = EmbeddingModel.from_registry("bge-small")


def get_json_files(directory: str) -> list[str]:
    json_files: list[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def embed(dataset_path: str, output_path: str) -> None:
    json_files = get_json_files(dataset_path)
    problems: list[str] = []
    for file_path in tqdm(json_files):
        with open(file_path, "r") as f:
            example = json.load(f)
        problems.append(example["problem"])
    embs = model.encode(problems).tolist()

    emb_dict = {
        os.path.relpath(file_path, dataset_path): emb
        for file_path, emb in zip(json_files, embs)
    }

    with open(output_path, "w") as f:
        json.dump(emb_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM on MATH problems")
    _ = parser.add_argument(
        "dataset_path", type=str, help="Example: _datasets/MATH/test"
    )
    _ = parser.add_argument("output_path", type=str, help="Example: embeddings.json")
    args = parser.parse_args()

    embed(args.dataset_path, args.output_path)
