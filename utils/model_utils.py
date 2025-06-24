import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import numpy as np

def save_results(model, test_y, test_prediction, seed):
    if not os.path.exists('results'):
        os.makedirs('results')
    results_df = pd.DataFrame({'True Values': test_y, 'Predictions': test_prediction})
    results_df.to_csv(f'results/predictions_seed{seed}.csv', index=False)
    if hasattr(model, 'feature_importances_'):
        importances_df = pd.DataFrame({'Feature': range(model.n_features_in_), 'Importance': model.feature_importances_})
        importances_df.to_csv(f'results/feature_importance_seed{seed}.csv', index=False)