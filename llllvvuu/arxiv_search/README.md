Offline[^1] arXiv abstract search app, with hybrid search (BM25[^2] + vector search) and reranking.

[^1]: The link to the paper is external, so an Internet connection is required to follow it.
[^2]: For BM25, ideally we would have the full paper text. I haven't done this yet as it's quite resource intensive to acquire.

## Usage
```sh
pip install -r requirements.txt  # preferably in a virtual environment
python filter_categories.py
python embed.py  # might take a few hours on a laptop
python bm25.py
python search.py
```
