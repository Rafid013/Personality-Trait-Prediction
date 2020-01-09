import pandas as pd
from keras import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from process_status import calculate_max_length, clean_status
from lstm_model1 import define_model


traits = ['AGR', 'CON', 'EXT', 'NEU', 'OPN']
batch_size = 512
epochs = 20


if __name__ == '__main__':
    data = pd.read_csv('mypersonality.csv', delimiter=',', encoding='iso-8859-1')
    statuses = data['STATUS']
    clean_status(statuses)
    data.to_csv('cleaned_mypersonality.csv', sep=',')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(statuses)
    vocab_size = len(tokenizer.word_index) + 1

    max_length = calculate_max_length(statuses)

    model_s = define_model(vocab_size, 406, is_regression=True)
    model_c = define_model(vocab_size, 406, is_regression=False)

    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    fold = 1
    for train_idx, test_idx in kfold.split(data):
        train_x, test_x = data['STATUS'][train_idx], data['STATUS'][test_idx]
        train_x = tokenizer.texts_to_sequences(train_x)
        train_x = pad_sequences(train_x, maxlen=max_length)
        test_x = tokenizer.texts_to_sequences(test_x)
        test_x = pad_sequences(test_x, maxlen=max_length)

        for trait in traits:
            train_cy = data['c' + trait][train_idx]
            test_cy = data['c' + trait][test_idx]

            train_sy = data['s' + trait][train_idx]
            test_sy = data['s' + trait][test_idx]

            model_c.fit(train_x, train_cy, epochs=epochs, batch_size=batch_size)
            lstm_layer_model = Model(inputs=model_c.inputs, outputs=model_c.get_layer('lstm_layer').output)
            train_cx, test_cx = lstm_layer_model.predict(train_x), lstm_layer_model.predict(test_x)

            model_s.fit(train_x, train_sy, epochs=epochs, batch_size=batch_size)
            lstm_layer_model = Model(inputs=model_s.inputs, outputs=model_s.get_layer('lstm_layer').output)
            train_sx, test_sx = lstm_layer_model.predict(train_x), lstm_layer_model.predict(test_x)

            pd.DataFrame(train_cx).to_csv('Folds/train_c_' + trait + '_' + str(fold) + '.csv', sep=',', header=False)
            pd.DataFrame(train_sx).to_csv('Folds/train_s_' + trait + '_' + str(fold) + '.csv', sep=',', header=False)
            pd.DataFrame(test_cx).to_csv('Folds/test_c_' + trait + '_' + str(fold) + '.csv', sep=',', header=False)
            pd.DataFrame(test_sx).to_csv('Folds/test_s_' + trait + '_' + str(fold) + '.csv', sep=',', header=False)
