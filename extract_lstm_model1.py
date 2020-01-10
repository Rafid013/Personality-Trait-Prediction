from keras import Input, Model
from keras.layers import Embedding, Dropout, LSTM, Dense
import pandas as pd


traits = ['AGR', 'CON', 'EXT', 'NEU', 'OPN']
BATCH_SIZE = 512
EPOCHS = 20
VOCAB_SIZE = 16209
MAX_LENGTH = 406


def define_model(vocab_size, max_length, is_regression):
    inputs = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 512, mask_zero=True)(inputs)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(512, return_sequences=False, name='lstm_layer')(se2)
    if is_regression:
        outputs = Dense(1, activation='relu')(se3)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer='adam')
    else:
        outputs = Dense(1, activation='softmax')(se3)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


for fold in range(1, 11):
    train_x = pd.read_csv('Folds/train_x_' + str(fold) + '.csv', delimiter=',', header=None)
    test_x = pd.read_csv('Folds/test_x_' + str(fold) + '.csv', delimiter=',', header=None)
    for trait in traits:
        train_cy = pd.read_csv('Folds/train_cy_' + trait + '_' + str(fold) + '.csv', sep=',')
        train_sy = pd.read_csv('Folds/train_sy_' + trait + '_' + str(fold) + '.csv', sep=',')
        test_cy = pd.read_csv('Folds/test_cy_' + trait + '_' + str(fold) + '.csv', sep=',')
        test_sy = pd.read_csv('Folds/test_sy_' + trait + '_' + str(fold) + '.csv', sep=',')

        model_c = define_model(VOCAB_SIZE, MAX_LENGTH, is_regression=False)
        model_c.fit(train_x, train_cy, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        lstm_layer_model = Model(inputs=model_c.inputs, outputs=model_c.get_layer('lstm_layer').output)
        train_cx, test_cx = lstm_layer_model.predict(train_x), lstm_layer_model.predict(test_x)

        model_s = define_model(VOCAB_SIZE, MAX_LENGTH, is_regression=True)
        model_s.fit(train_x, train_sy, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        lstm_layer_model = Model(inputs=model_s.inputs, outputs=model_s.get_layer('lstm_layer').output)
        train_sx, test_sx = lstm_layer_model.predict(train_x), lstm_layer_model.predict(test_x)

        pd.DataFrame(train_cx).to_csv('Folds/train_c_features_' + trait + '_' + str(fold) + '.csv', sep=',',
                                      header=False)
        pd.DataFrame(train_sx).to_csv('Folds/train_s_features_' + trait + '_' + str(fold) + '.csv', sep=',',
                                      header=False)
        pd.DataFrame(test_cx).to_csv('Folds/test_c_features_' + trait + '_' + str(fold) + '.csv', sep=',', header=False)
        pd.DataFrame(test_sx).to_csv('Folds/test_s_features_' + trait + '_' + str(fold) + '.csv', sep=',', header=False)
