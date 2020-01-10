from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


traits = ['AGR', 'CON', 'EXT', 'NEU', 'OPN']

result = pd.DataFrame()
result['TRAITS'] = pd.Series(traits)

accuracies = []

for trait in traits:
    predictions = []
    true_ys = []
    for fold in range(1, 11):
        prediction = np.genfromtxt('Predictions/svm_classifier_lstm1_prediction_c_' + trait + '_' + str(fold) + '.csv')\
            .tolist()
        predictions += prediction

        true_y = np.genfromtxt('Folds/test_cy_' + trait + '_' + str(fold) + '.csv', skip_header=1).tolist()
        true_ys += true_y
    acc = accuracy_score(true_ys, predictions)
    accuracies.append(acc)

result['ACCURACY'] = pd.Series(accuracies)

result.to_csv('Results/result_svm_classifier_lstm1.csv', sep=',', index=False)
