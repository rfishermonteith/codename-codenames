import json
import pickle
from pathlib import Path

import fire
import numpy as np

from sklearn.metrics import pairwise_distances


def main(embedding_type: str = "hash", size: str = "big", file_format: str = "json"):
    """Calculate pairwise distances for all dictionary words and Codenames words"""

    # Load embeddings and Codenames words
    embedding_file = f"data/embeddings/{embedding_type}_{size}.json"
    with open(embedding_file, "r") as f:
        embeddings = json.load(f)

    with open("data/wordlists/codenames-wordlist-eng.txt", "r") as f:
        code_name_words = f.read().lower().split("\n")

    a = np.array([embeddings[x] for x in code_name_words])
    b = np.array([embeddings[y] for y in embeddings.keys()])

    all_pairwise_distances = 1 - pairwise_distances(a, b, metric="cosine")
    # all_pairwise_distances = pairwise_distances(a, b, metric="euclidean")

    distance_file = f"data/distances/{embedding_type}_{size}.{file_format}"
    Path(distance_file).parent.mkdir(parents=True, exist_ok=True)

    distances_dict = {
        "row_indices": code_name_words,
        "col_indices": list(embeddings.keys()),
        "distances": all_pairwise_distances.tolist()
    }

    match file_format:
        case "json":
            with open(distance_file, "w") as f:
                json.dump(distances_dict, f)
        case "pkl":
            with open(distance_file, 'wb') as f:
                pickle.dump(distances_dict, f, protocol=-1)
        case _:
            raise ValueError(f'file_format "{file_format}" unknown')


if __name__ == "__main__":
    fire.Fire(main)
