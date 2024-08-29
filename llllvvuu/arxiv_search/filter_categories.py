import argparse
import json
import os
import logging


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter arXiv metadata by categories")
    parser.add_argument("input_file", help="Input JSONL file")
    parser.add_argument("output_dir", help="Output directory for filtered JSON files")
    parser.add_argument(
        "-c",
        "--categories",
        nargs="+",
        default=["cs.CV", "cs.LG", "cs.CL", "cs.AI", "cs.NE", "cs.RO"],
        help="Categories to include (default: cs.CV cs.LG cs.CL cs.AI cs.NE cs.RO)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Number of results per output file (default: 250)",
    )
    return parser.parse_args()


def filter_categories(
    input_file: str, output_dir: str, categories: list[str], chunk_size: int
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    filtered_entries = []
    file_counter = 0

    with open(input_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            entry_categories = entry.get("categories", "").split()
            if any(cat in entry_categories for cat in categories):
                filtered_entries.append(entry)

            if len(filtered_entries) >= chunk_size:
                save_chunk(filtered_entries, output_dir, file_counter)
                filtered_entries = []
                file_counter += 1

    if filtered_entries:
        save_chunk(filtered_entries, output_dir, file_counter)


def save_chunk(entries: list[dict], output_dir: str, file_counter: int) -> None:
    filename = f"{file_counter}.json"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        for entry in entries:
            json.dump(entry, f, separators=(",", ":"))
            f.write("\n")
    logging.info(f"Saved {filename}")


def main() -> None:
    args = parse_arguments()
    filter_categories(
        args.input_file, args.output_dir, args.categories, args.chunk_size
    )


if __name__ == "__main__":
    main()
