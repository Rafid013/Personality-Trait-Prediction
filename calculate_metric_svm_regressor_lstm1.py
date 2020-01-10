from scipy.stats import pearsonr
import numpy as np
import pandas as pd


traits = ['AGR', 'CON', 'EXT', 'NEU', 'OPN']

result = pd.DataFrame()
result['TRAITS'] = pd.Series(traits)

pearson_correlations = []

for trait in traits:
    predictions = []
    true_ys = []
    for fold in range(1, 11):
        prediction = np.genfromtxt('Predictions/svm_regressor_lstm1_prediction_s_' + trait + '_' + str(fold) + '.csv')\
            .tolist()
        predictions += prediction

        true_y = np.genfromtxt('Folds/test_sy_' + trait + '_' + str(fold) + '.csv', skip_header=1).tolist()
        true_ys += true_y
    corr, _ = pearsonr(true_ys, predictions)
    pearson_correlations.append(corr)

result['PEARSON CORRELATION'] = pd.Series(pearson_correlations)

result.to_csv('Results/result_svm_regressor_lstm1.csv', sep=',', index=False)
