from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


traits = ['AGR', 'CON', 'EXT', 'NEU', 'OPN']

steps = [('standard_scaling', StandardScaler()),
         ('SVM', SVC(gamma='auto', random_state=1, verbose=True, cache_size=3072))]


for fold in range(1, 11):
    for trait in traits:
        train_x = pd.read_csv('LSTM 1 Features/train_c_features_' + trait + '_' + str(fold) + '.csv', delimiter=',',
                              header=None).iloc[:, 1:]
        test_x = pd.read_csv('LSTM 1 Features/test_c_features_' + trait + '_' + str(fold) + '.csv', delimiter=',',
                             header=None).iloc[:, 1:]

        train_ftr = pd.read_csv('Folds/train_features_' + str(fold) + '.csv', delimiter=',', header=None)
        test_ftr = pd.read_csv('Folds/test_features_' + str(fold) + '.csv', delimiter=',', header=None)

        train_x = pd.concat([train_x, train_ftr], axis=1, sort=False, ignore_index=True)
        test_x = pd.concat([test_x, test_ftr], axis=1, sort=False, ignore_index=True)

        train_y = pd.read_csv('Folds/train_cy_' + trait + '_' + str(fold) + '.csv', delimiter=',')

        pipeline = Pipeline(steps=steps)

        pipeline.fit(train_x, train_y)

        predictions = pipeline.predict(test_x)

        pd.DataFrame(predictions).to_csv('Predictions/svm_classifier_lstm1_prediction_c_' + trait + '_' +
                                         str(fold) + '.csv', sep=',', header=False, index=False)
