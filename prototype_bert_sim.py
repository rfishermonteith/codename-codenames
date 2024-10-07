import json

import fire
from scipy.spatial.distance import cosine

import random

from tqdm import tqdm


def similarity(vec_1, vec_2):
    return 1 - cosine(vec_1, vec_2)


def main(embeddings_file: str = "embeddings.json"):
    # Load the vectors
    with open(embeddings_file, "r") as f:
        embeddings = json.load(f)

    # Randomly generate a board
    with open("data/wordlists/codenames-wordlist-eng.txt", "r") as f:
        words = f.read().lower().split("\n")

    # random.seed(52)
    sampled_words = random.sample(words, 19)

    remaining_words = [x for x in embeddings.keys() if x not in sampled_words]
    my_words = sampled_words[:9]
    their_words = sampled_words[9:18]
    assassin = sampled_words[-1]

    # Find the remaining word which is closest to my words, but far from their words and the assassin
    pairwise_distances = {}
    for y in tqdm(remaining_words):
        pairwise_distances[y] = {}
        for x in sampled_words:
            pairwise_distances[y][x] = similarity(embeddings[x], embeddings[y])

    # Score each word based on how many guesses we'd get correct
    scores = {}
    for y in remaining_words:
        word_scores = [pairwise_distances[y][x] for x in sampled_words]
        word_selections = sampled_words

        sorted_words = sorted(zip(word_selections, word_scores), key=lambda x: x[1], reverse=True)

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

    print(f"best word: {best_word}")
    print(f"score: {scores[best_word]}")


if __name__ == "__main__":
    fire.Fire(main)
