import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from safetensors.numpy import save_file
import time


def parse_arguments():
    parser = argparse.ArgumentParser(description="Embed filtered arXiv metadata chunks")
    parser.add_argument(
        "model_name",
        type=str,
        help="HuggingFace model name (e.g., 'BAAI/bge-small-en-v1.5')",
    )
    parser.add_argument(
        "input_dir", type=str, help="Input directory containing filtered JSONL chunks"
    )
    parser.add_argument("output_dir", type=str, help="Output directory for embeddings")
    return parser.parse_args()


def load_json_chunk(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def embed_texts(model, tokenizer, texts, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def process_chunks(model, tokenizer, input_dir, output_dir, device):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_dir.glob("*.json"):
        output_file = output_dir / f"{input_file.stem}.safetensors"

        if output_file.exists():
            print(f"Skipping {input_file.name} - {output_file.name} already exists")
            continue

        start_time = time.perf_counter()

        chunk_data = load_json_chunk(input_file)
        abstracts = [
            item.get("title", "") + "[SEP]" + item.get("abstract", "")
            for item in chunk_data
        ]

        embeddings = embed_texts(model, tokenizer, abstracts, device)

        save_file({"embeddings": embeddings}, str(output_file))

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        print(
            f"Processed {input_file.name} -> {output_file.name} in {processing_time:.2f} seconds"
        )


def main():
    args = parse_arguments()

    print("Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    print("Processing chunks...")
    start_time = time.perf_counter()
    process_chunks(model, tokenizer, args.input_dir, args.output_dir, device)
    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(f"Embedding complete. Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
