import numpy as np


def _get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def train_dict_embedding_matrix(embedding_file, tokenizer, max_features, embed_size):

    embeddings_index = dict(_get_coefs(*o.strip().split()) for o in open(embedding_file, encoding='utf-8'))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
