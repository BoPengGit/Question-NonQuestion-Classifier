import argparse, string, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utils.models import GRU_model, LSTM_model
from utils.train_utils import train_dict_embedding_matrix


def main():
    parser = argparse.ArgumentParser(description='GRU and LSTM model for classifying if a sentence is a question.')

    parser.add_argument('train_file_path')
    parser.add_argument('embedding_path')
    parser.add_argument('-bs', '--batch-size', type=int, default=1000)  # training batch size
    parser.add_argument('-sl', '--sentence-length', type=int, default=25)  # max word length of sentences
    parser.add_argument('-es', '--embed-size', type=int)  # dimension of word vectors i.e. (25, 50, 300)
    parser.add_argument('-mw', '--max-features', type=int, default=30000)  # max number of tokens in dictionary used
    parser.add_argument('-ep', '--epochs', type=int, default=2)  # number of epochs for training
    parser.add_argument('-gm', '--gru-save-model', type=str, default='GRU.h5')
    parser.add_argument('-lm', '--lstm-save-model', type=str, default='LSTM.h5')

    args = parser.parse_args()

    embedding_file = args.embedding_path

    embed_size = args.embed_size
    max_features = args.max_features
    maxlen = args.sentence_length

    print('Loading data')
    data = pd.read_csv(args.train_file_path)

    data['questions'] = data['questions'].fillna('_na_').values
    data['non-questions'] = data['non-questions'].fillna('_na_').values

    x_data = np.concatenate((data['questions'], data['non-questions']), axis=0)
    x_data_no_punc = np.array([''.join((char for char in sentence if char not in string.punctuation)) for sentence in x_data])
    y_data = np.concatenate((np.zeros(len(data['questions'])), np.ones(len(data['non-questions']))), axis=0)

    print('Tokenizing data')
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(x_data_no_punc))
    list_tokenized_train = tokenizer.texts_to_sequences(x_data_no_punc)
    X_tr = pad_sequences(list_tokenized_train, maxlen=maxlen)

    print('Creating embedding matrix for training tokens.')
    embedding_matrix = train_dict_embedding_matrix(embedding_file, tokenizer, max_features, embed_size)

    print('Training GRU model')
    GRU = GRU_model(maxlen, max_features, embed_size, embedding_matrix)

    GRU.fit(X_tr, y_data, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.2)

    print('Training LSTM model')
    LSTM = LSTM_model(maxlen, max_features, embed_size, embedding_matrix)

    LSTM.fit(X_tr, y_data, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.2)

    print('Saving trained models')
    GRU.save(args.gru_save_model)
    LSTM.save(args.lstm_save_model)


if __name__ == '__main__':
    main()
