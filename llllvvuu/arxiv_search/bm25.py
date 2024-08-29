import argparse
import os
import json
import logging
import re
import string
from collections import defaultdict
import math
import msgpack

STOP_WORDS = set(
    ["the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "is", "are"]
)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return [word for word in text.split() if word not in STOP_WORDS]


def process_jsonl_file(
    file_path: str, word_doc_freq: dict[str, dict[int, int]], doc_lengths: list[int]
) -> None:
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                doc = json.loads(line.strip())
                doc_id = len(doc_lengths)
                title: str = doc.get("title", "")
                abstract: str = doc.get("abstract", "")

                words = tokenize(title + " " + abstract)
                doc_lengths.append(len(words))

                unique_words = set(words)
                for word in unique_words:
                    word_doc_freq[word][doc_id] += 1

            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON line in {file_path}")


def calculate_idf(
    word_doc_freq: dict[str, dict[int, int]], total_docs: int
) -> dict[str, float]:
    return {
        word: math.log(total_docs / len(docs)) for word, docs in word_doc_freq.items()
    }


def main(directory: str, output: str, verbose: bool) -> None:
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    word_doc_freq: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    doc_lengths: list[int] = []

    files = sorted(
        [f for f in os.listdir(directory) if f.endswith((".json", ".jsonl"))]
    )

    for filename in files:
        file_path = os.path.join(directory, filename)
        process_jsonl_file(file_path, word_doc_freq, doc_lengths)
        logging.info(f"Processed {filename}")

    total_docs = len(doc_lengths)
    avg_doc_length = sum(doc_lengths) / total_docs if total_docs > 0 else 0
    idf_scores = calculate_idf(word_doc_freq, total_docs)

    output_data = {
        "word_doc_frequencies": {
            word: dict(docs) for word, docs in word_doc_freq.items()
        },
        "doc_lengths": doc_lengths,
        "average_doc_length": avg_doc_length,
        "idf_scores": idf_scores,
    }

    with open(output, "wb") as outfile:
        packed = msgpack.packb(output_data)
        outfile.write(packed)

    logging.info(f"Output written to {output}")
    print(f"Number of unique words: {len(word_doc_freq)}")
    print(f"Number of documents processed: {total_docs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process JSONL files, count unique words, and output statistics in msgpack format."
    )
    parser.add_argument(
        "directory", help="Path to the directory containing JSONL chunks"
    )
    parser.add_argument("output", help="Path to the output msgpack file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )

    args = parser.parse_args()

    main(args.directory, args.output, args.verbose)
