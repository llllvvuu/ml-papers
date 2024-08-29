import argparse
import json
import msgpack
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from safetensors import safe_open
from tqdm import tqdm
import re
import string

STOP_WORDS = set(
    ["the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "is", "are"]
)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return [word for word in text.split() if word not in STOP_WORDS]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Perform nearest neighbor search over embeddings and BM25 search"
    )
    parser.add_argument(
        "model_name", type=str, help="Name of the sentence transformer model"
    )
    parser.add_argument("jsonl_dir", type=str, help="Directory containing JSONL chunks")
    parser.add_argument(
        "embeddings_dir", type=str, help="Directory containing embedded tensors"
    )
    parser.add_argument("bm25_msgpack", type=str, help="Path to the BM25 msgpack file")
    parser.add_argument("query", type=str, help="Query text for search")
    parser.add_argument(
        "--num-vector-results",
        type=int,
        default=5,
        help="Number of vector search results to return",
    )
    parser.add_argument(
        "--num-bm25-results",
        type=int,
        default=5,
        help="Number of BM25 search results to return",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        help="Name of the reranker model to use (e.g., 'BAAI/bge-reranker-v2-m3')",
    )
    return parser.parse_args()


def load_embeddings(embeddings_dir):
    embeddings_dir = Path(embeddings_dir)
    all_embeddings = []
    file_mapping = {}
    current_index = 0

    for file_path in tqdm(
        list(embeddings_dir.glob("*.safetensors")), desc="Loading embeddings"
    ):
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            embeddings = f.get_tensor("embeddings")
        num_embeddings = embeddings.shape[0]
        all_embeddings.append(embeddings)
        file_mapping[file_path.stem] = (current_index, current_index + num_embeddings)
        current_index += num_embeddings

    return torch.cat(all_embeddings, dim=0), file_mapping


def cosine_similarity(x1, x2):
    return torch.nn.functional.cosine_similarity(x1, x2, dim=1)


def nearest_neighbor_search(query_embedding, embeddings, k):
    similarities = cosine_similarity(query_embedding, embeddings)
    top_k_values, top_k_indices = torch.topk(similarities, k)
    return top_k_indices, top_k_values


def load_bm25_data(bm25_msgpack):
    with open(bm25_msgpack, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False, strict_map_key=False)


def bm25_search(query, bm25_data, k):
    word_doc_freq = bm25_data["word_doc_frequencies"]
    doc_lengths = bm25_data["doc_lengths"]
    avg_doc_length = bm25_data["average_doc_length"]
    idf_scores = bm25_data["idf_scores"]

    query_terms = tokenize(query)
    scores = [0] * len(doc_lengths)

    k1 = 1.5
    b = 0.75

    for term in query_terms:
        if term in word_doc_freq:
            idf = idf_scores[term]
            for doc_id, freq in word_doc_freq[term].items():
                doc_id = int(doc_id)
                numerator = idf * freq * (k1 + 1)
                denominator = freq + k1 * (
                    1 - b + b * doc_lengths[doc_id] / avg_doc_length
                )
                scores[doc_id] += numerator / denominator

    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :k
    ]
    top_k_scores = [scores[i] for i in top_k_indices]

    return top_k_indices, top_k_scores


def print_result(i, similarity, result, search_type):
    print(f"{search_type} Result {i+1} (Score: {similarity:.4f}):")
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"URL: https://arxiv.org/abs/{result.get('id', 'N/A')}")
    print(f"Authors: {result.get('authors', 'N/A')}")
    print(f"Last Updated: {result.get('update_date', 'N/A')}")
    print(f"Abstract: {result.get('abstract', 'N/A')}")
    print()


def count_lines_in_files(jsonl_dir):
    jsonl_dir = Path(jsonl_dir)
    file_line_counts = {}
    total_lines = 0

    for file_path in tqdm(
        sorted(list(jsonl_dir.glob("*.json"))), desc="Counting lines"
    ):
        with open(file_path, "r") as f:
            line_count = sum(1 for _ in f)
        file_line_counts[file_path.stem] = (total_lines, total_lines + line_count)
        total_lines += line_count

    return file_line_counts, total_lines


def load_reranker(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def rerank_results(query, results, reranker_tokenizer, reranker_model):
    pairs = [
        [query, f"{result["title"]}[SEP]{result["abstract"]}"] for result in results
    ]
    with torch.no_grad():
        inputs = reranker_tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        scores = (
            reranker_model(**inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
        )

    reranked_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return zip(*reranked_results)


def main():
    args = parse_arguments()

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    reranker_tokenizer, reranker_model = None, None
    if args.reranker:
        print(f"Loading reranker model: {args.reranker}")
        reranker_tokenizer, reranker_model = load_reranker(args.reranker)
        reranker_model = reranker_model.to(device)

    print("Loading embeddings...")
    embeddings, file_mapping = load_embeddings(args.embeddings_dir)
    embeddings = embeddings.to(device)

    print("Loading BM25 data...")
    bm25_data = load_bm25_data(args.bm25_msgpack)

    print("Counting lines in JSONL files...")
    file_line_counts, total_lines = count_lines_in_files(args.jsonl_dir)

    print("Encoding query...")
    inputs = tokenizer(
        args.query, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state[:, 0, :]

    print("Performing nearest neighbor search...")
    top_k_indices, top_k_values = nearest_neighbor_search(
        query_embedding, embeddings, args.num_vector_results
    )

    print("Performing BM25 search...")
    bm25_top_k_indices, bm25_top_k_scores = bm25_search(
        args.query, bm25_data, args.num_bm25_results
    )

    vector_results = []
    bm25_results = []

    if not args.reranker:
        print(
            f"\nTop {args.num_vector_results} vector search results for query: '{args.query}'\n"
        )
    for i, (index, similarity) in enumerate(zip(top_k_indices, top_k_values)):
        for file_name, (start_index, end_index) in file_mapping.items():
            if start_index <= index < end_index:
                relative_index = index - start_index
                break

        input_file = Path(args.jsonl_dir) / f"{file_name}.json"
        with open(input_file, "r") as f:
            for j, line in enumerate(f):
                if j == relative_index:
                    result = json.loads(line)
                    vector_results.append(result)
                    if not args.reranker:
                        print_result(i, similarity, result, "Vector")
                    break

    if not args.reranker:
        print(
            f"\nTop {args.num_bm25_results} BM25 search results for query: '{args.query}'\n"
        )
    for i, (index, score) in enumerate(zip(bm25_top_k_indices, bm25_top_k_scores)):
        for file_name, (start_index, end_index) in file_line_counts.items():
            if start_index <= index < end_index:
                relative_index = index - start_index
                break

        input_file = Path(args.jsonl_dir) / f"{file_name}.json"
        with open(input_file, "r") as f:
            for j, line in enumerate(f):
                if j == relative_index:
                    result = json.loads(line)
                    bm25_results.append(result)
                    if not args.reranker:
                        print_result(i, score, result, "BM25")
                    break

    if args.reranker:
        print("\nReranking results...")
        reranked_results, reranker_scores = rerank_results(
            args.query,
            vector_results + bm25_results,
            reranker_tokenizer,
            reranker_model,
        )

        print(f"\nTop {args.num_vector_results} reranked search results:\n")
        for i, (result, score) in enumerate(zip(reranked_results, reranker_scores)):
            print_result(i, score.item(), result, "Reranked")


if __name__ == "__main__":
    main()
