import pickle, string, argparse, numpy as np, pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


def main():
    parser = argparse.ArgumentParser(
        description="Predict question/non-question classification on sentences data using pretrained model.")

    parser.add_argument('sentences_data_file_path')
    parser.add_argument('-tp', '--tokenizer-path', type=str, default='utils/pretrained files/tokenizer.pickle')
    parser.add_argument('-gm', '--GRU-pretrained-model-path', type=str, default='utils/pretrained files/GRU.h5')
    parser.add_argument('-lm', '--LSTM-pretrained-model-path', type=str, default='utils/pretrained files/LSTM.h5')
    parser.add_argument('-op', '--output-file-path', type=str, default='predicted_results.csv')

    args = parser.parse_args()

    print('Loading data.')
    try:
        data = pd.read_csv(args.sentences_data_file_path)
    except UnicodeDecodeError:
        data = pd.read_csv(args.sentences_data_file_path, encoding='cp1252')

    with open(args.tokenizer_path, 'rb') as input_file:
        tokenizer = pickle.load(input_file)

    GRU = load_model(args.GRU_pretrained_model_path)
    LSTM = load_model(args.LSTM_pretrained_model_path)

    data_no_punc = np.array([''.join((char for char in sentence if char not in string.punctuation))
                             for sentence in data['sentences']])

    print('Tokenizing sentences.')
    list_tokenized_data = tokenizer.texts_to_sequences(data_no_punc)
    tokenized_pad_data = pad_sequences(list_tokenized_data, maxlen=25)

    print('Making predictions on LSTM model.')
    LSTM_predicted = LSTM.predict(tokenized_pad_data, verbose=1)

    print('Making predictions on GRU model.')
    GRU_predicted = GRU.predict(tokenized_pad_data, verbose=1)

    weighted_prediction = (GRU_predicted + LSTM_predicted) / 2
    data['prediction'] = weighted_prediction.round()

    print('Outputting predictions file to csv.')
    data.to_csv(args.output_file_path, index=False)


if __name__ == '__main__':
    main()
