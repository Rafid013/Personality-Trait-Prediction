import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from process_status import calculate_max_length, clean_status


traits = ['AGR', 'CON', 'EXT', 'NEU', 'OPN']

if __name__ == '__main__':
    data = pd.read_csv('mypersonality.csv', delimiter=',', encoding='iso-8859-1')
    statuses = data['STATUS']
    clean_status(statuses)
    data.to_csv('cleaned_mypersonality.csv', sep=',')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(statuses)
    vocab_size = len(tokenizer.word_index) + 1

    max_length = calculate_max_length(statuses)

    print(max_length)
    print(vocab_size)

    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    fold = 1
    for train_idx, test_idx in kfold.split(data):
        train_x, test_x = data['STATUS'][train_idx], data['STATUS'][test_idx]
        train_x = tokenizer.texts_to_sequences(train_x)
        train_x = pad_sequences(train_x, maxlen=max_length)
        test_x = tokenizer.texts_to_sequences(test_x)
        test_x = pad_sequences(test_x, maxlen=max_length)

        pd.DataFrame(train_x).to_csv('Folds/train_x_' + str(fold) + '.csv', sep=',', index=False, header=False)
        pd.DataFrame(test_x).to_csv('Folds/test_x_' + str(fold) + '.csv', sep=',', index=False, header=False)

        for trait in traits:
            train_cy = data['c' + trait][train_idx]
            test_cy = data['c' + trait][test_idx]

            train_sy = data['s' + trait][train_idx]
            test_sy = data['s' + trait][test_idx]

            pd.DataFrame(train_cy).to_csv('Folds/train_cy_' + trait + '_' + str(fold) + '.csv', sep=',', index=False)
            pd.DataFrame(train_sy).to_csv('Folds/train_sy_' + trait + '_' + str(fold) + '.csv', sep=',', index=False)
            pd.DataFrame(test_cy).to_csv('Folds/test_cy_' + trait + '_' + str(fold) + '.csv', sep=',', index=False)
            pd.DataFrame(test_sy).to_csv('Folds/test_sy_' + trait + '_' + str(fold) + '.csv', sep=',', index=False)
        fold += 1
