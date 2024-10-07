import json
import pickle

import fire

import random

import numpy as np

from tqdm import tqdm


def main(embedding_type: str = "hash", size: str = "big", file_format: str = "json", n_games: int = 10):
    """
    Plays n_games games and saves the score for each game
    """

    print("Starting")
    # Load the vectors
    distance_file = f"data/distances/{embedding_type}_{size}.{file_format}"

    match file_format:
        case "json":
            with open(distance_file, "r") as f:
                distances = json.load(f)
        case "pkl":
            with open(distance_file, 'rb') as f:
                distances = pickle.load(f)
        case _:
            raise ValueError(f'file_format "{file_format}" unknown')

    print("Finished loading")

    d = np.array(distances['distances'])

    codename_words = distances["row_indices"]
    dictionary_words = distances['col_indices']

    game_scores = []

    for _ in tqdm(range(n_games)):

        # random.seed(52)
        sampled_words = random.sample(codename_words, 19)

        remaining_words = [x for x in dictionary_words if x not in sampled_words]
        my_words = sampled_words[:9]
        their_words = sampled_words[9:18]
        assassin = sampled_words[-1]

        # Score each word based on how many guesses we'd get correct
        scores = {}
        row_inds = [codename_words.index(x) for x in sampled_words]
        col_inds = [ind for ind, w in enumerate(dictionary_words) if w not in sampled_words]
        # col_inds = [dictionary_words.index(y) for y in remaining_words]

        possible_distances = d[row_inds][:, col_inds]

        for ind, y in enumerate(remaining_words):
            # row_inds = 1
            # word_scores = [d[codename_words.index(x)][dictionary_words.index(y)] for x in sampled_words]
            word_scores = possible_distances[:, ind]
            # word_selections = sampled_words

            sorted_words = sorted(zip(sampled_words, word_scores), key=lambda x: x[1], reverse=True)

            # Calculate the score of this word
            score = 0
            for s in sorted_words:
                if s[0] not in my_words:
                    break
                else:
                    score += 1

            scores[y] = score

        # Find the max score
        best_word = max(scores, key=scores.get)

        # print(f"best word: {best_word}")
        # print(f"score: {scores[best_word]}")
        game_scores.append(scores[best_word])

    # Write the results out
    filename = f"data/outputs/{embedding_type}_{size}_{n_games:08}.json"

    with open(filename, "w") as f:
        json.dump(game_scores, f)


if __name__ == "__main__":
    fire.Fire(main)