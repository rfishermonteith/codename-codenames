import json

import fire
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

from scipy.spatial.distance import cosine





def get_batch_embeddings_bert(texts):
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


    import hashlib

    return [list(hashlib.sha3_512(x.encode()).digest()) + list(hashlib.md5(x.encode()).digest()) for x in texts]


def main(embedding_type: str = "bert"):
    # Load the wordlist

    with open('data/wordlists/codenames-wordlist-eng.txt', 'r') as f:
        word_list_codenames = f.read().lower().split('\n')

    with open('data/wordlists/dictionary-wordlist.txt', 'r') as f:
        word_list_dict = f.read().lower().split('\n')

    word_list = word_list_codenames + word_list_dict

    mapping = {}

    match embedding_type:
        case "bert":

            all_embeddings = get_batch_embeddings_bert(word_list)
            output_filename = "embeddings_bert.json"
        case "hash":
            all_embeddings = get_batch_embeddings_hash(word_list)
            output_filename = "embeddings_hash.json"

        case _:
            raise ValueError("embedding_type unknown")

    mapping = {k: v for k, v in zip(word_list, all_embeddings)}

    with open(output_filename, "w") as f:
        json.dump(mapping, f)


if __name__ == "__main__":
    fire.Fire(main)


