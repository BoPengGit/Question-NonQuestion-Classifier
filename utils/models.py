from keras.layers import Dense, Embedding, Input
from keras.layers import Bidirectional, Dropout, CuDNNGRU, LSTM, GlobalMaxPool1D
from keras.models import Model
from keras.optimizers import RMSprop


def GRU_model(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(x)
    x = Dropout(0.1)(x)
    x = Bidirectional(CuDNNGRU(50, return_sequences=False))(x)
    x = Dense(50, activation='relu')(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['accuracy'])

    return model


def LSTM_model(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.1)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
