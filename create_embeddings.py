import json
import random

import fire
from pathlib import Path


def get_batch_embeddings_bert(texts):
    from transformers import AutoTokenizer, AutoModel
    import torch

    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    # Tokenize the texts and convert them to tensor format
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    # Get the model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

    # Mean pooling over the sequence length dimension
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * attention_mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask

    return mean_pooled.numpy().tolist()


def get_batch_embeddings_hash(texts):
    """Create hashed embeddings"""
    import hashlib

    hashes = [
        hashlib.blake2b,
        hashlib.blake2s,
        hashlib.md5,
        # hashlib.scrypt,
        hashlib.sha1,
        hashlib.sha224,
        hashlib.sha256,
        hashlib.sha384,
        hashlib.sha3_224,
        hashlib.sha3_256,
        hashlib.sha3_384,
        hashlib.sha3_512,
        hashlib.sha512,
        # hashlib.shake_128,
        # hashlib.shake_256
    ]

    applied = []
    for x in texts:
        applied_hashes = [list(h(x.encode()).digest()) for h in hashes]
        applied_hashes.extend([list(h(x[::-1].encode()).digest()) for h in hashes])
        applied_hashes.extend([list(h(x[1::].encode()).digest()) for h in hashes])
        applied_hashes.extend([list(h(x[::-2].encode()).digest()) for h in hashes])
        to_append = []
        print(len(applied_hashes[-1]))
        for ah in applied_hashes:
            to_append.extend(ah)
        applied.append(to_append)
        print(len(to_append))

    return applied


def main(embedding_type: str = "bert", size: str = "big"):
    """
    Create embeddings for all words in game, and dictionary
    """

    # Load the Codenames wordlist
    with open('data/wordlists/codenames-wordlist-eng.txt', 'r') as f:
        word_list_codenames = f.read().lower().split('\n')

    # Load the dictionary wordlist
    with open('data/wordlists/dictionary-wordlist.txt', 'r') as f:
        word_list_dict = f.read().lower().split('\n')

    if size == "small":
        word_list_dict = random.sample(word_list_dict, 1000)
    if size not in ["small", "big"]:
        raise ValueError(f"size '{size}' unknown")

    word_list = word_list_codenames + word_list_dict

    match embedding_type:
        case "bert":
            all_embeddings = get_batch_embeddings_bert(word_list)
        case "hash":
            all_embeddings = get_batch_embeddings_hash(word_list)
        case _:
            raise ValueError(f"embedding_type '{embedding_type}' unknown")

    mapping = {k: v for k, v in zip(word_list, all_embeddings)}

    embedding_file = f"data/embeddings/{embedding_type}_{size}.json"
    Path(embedding_file).parent.mkdir(parents=True, exist_ok=True)
    with open(embedding_file, "w") as f:
        json.dump(mapping, f)


if __name__ == "__main__":
    fire.Fire(main)
