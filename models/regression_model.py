import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class RegressionModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        self.best_model = None
        self.all_labels = None
        self.train_y = None
        self.test_y = None
        self.test_y_original = None

    def fit_label_scaler(self, all_data):
        self.all_labels = all_data['label'].astype(float).values.reshape(-1, 1)
        self.label_scaler.fit(self.all_labels)

    def prepare_labels(self, all_data):
        train_mask = all_data['group'] == 'train'
        test_mask = all_data['group'] == 'test'
        self.train_y = self.label_scaler.transform(all_data[train_mask]['label'].values.reshape(-1, 1)).flatten()
        self.test_y = self.label_scaler.transform(all_data[test_mask]['label'].values.reshape(-1, 1)).flatten()
        self.test_y_original = all_data[test_mask]['label'].values.flatten()

    def scale_features(self, train_features, test_features):
        train_scaled = self.feature_scaler.fit_transform(train_features)
        test_scaled = self.feature_scaler.transform(test_features)
        return train_scaled, test_scaled

    def hyperparameter_tuning(self, X, y, param_grid):
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=self.random_state),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        self.best_model = grid_search.best_estimator_
        return grid_search

    def predict(self, X_test):
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        prediction_normalized = self.best_model.predict(X_test)
        prediction = self.label_scaler.inverse_transform(prediction_normalized.reshape(-1, 1)).flatten()
        return prediction

    def evaluate(self, y_pred):
        y_true_original = self.label_scaler.inverse_transform(self.test_y.reshape(-1, 1)).flatten()
        mse = mean_squared_error(y_true_original, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_original, y_pred)
        return {'MSE': mse, 'RMSE': rmse, 'R²': r2}