import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from process_status import calculate_max_length, clean_status
import numpy as np

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

    tokenized_statuses = tokenizer.texts_to_sequences(statuses)
    padded_statuses = pad_sequences(tokenized_statuses, maxlen=max_length)

    data_features = np.array(data[['NETWORKSIZE', 'BETWEENNESS', 'NBETWEENNESS', 'DENSITY', 'BROKERAGE',
                                   'NBROKERAGE', 'TRANSITIVITY']])

    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    fold = 1
    for train_idx, test_idx in kfold.split(data):

        train_x = padded_statuses[train_idx]
        test_x = padded_statuses[test_idx]

        train_features = pd.DataFrame(data_features[train_idx])
        test_features = pd.DataFrame(data_features[test_idx])

        pd.DataFrame(train_x).to_csv('Folds/train_x_' + str(fold) + '.csv', sep=',', index=False, header=False)
        pd.DataFrame(test_x).to_csv('Folds/test_x_' + str(fold) + '.csv', sep=',', index=False, header=False)

        pd.DataFrame(train_features).to_csv('Folds/train_features_' + str(fold) + '.csv', sep=',',
                                            index=False, header=False)
        pd.DataFrame(test_features).to_csv('Folds/test_features_' + str(fold) + '.csv', sep=',',
                                           index=False, header=False)

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
