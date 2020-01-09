from keras import Input, Model
from keras.layers import Embedding, Dropout, LSTM, Dense


def define_model(vocab_size, max_length, is_regression):
    inputs = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 1024, mask_zero=True)(inputs)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(1024, return_sequences=False, name='lstm_layer')(se2)
    if is_regression:
        outputs = Dense(1, activation='relu')(se3)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer='adam')
    else:
        outputs = Dense(1, activation='softmax')(se3)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam')
    return model
