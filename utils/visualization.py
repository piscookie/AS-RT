import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style('whitegrid')

def plot_feature_importance(model, feature_names, top_n=20):
    import numpy as np
    print(np.__version__)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predictions')
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Residual Distribution')
    plt.tight_layout()
    plt.show()

def plot_prediction_comparison(y_true, y_pred, mean_mse, mean_rmse, mean_r2):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', label='Predictions')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', linewidth=2, label='Ideal')
    errors = np.abs(y_pred - y_true)
    plt.errorbar(y_true, y_pred, yerr=errors, fmt='none', ecolor='gray', alpha=0.3)
    plt.xlabel('True Values (Original Scale)')
    plt.ylabel('Predictions (Original Scale)')
    plt.title('Prediction vs True Values')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.text(min(y_true), max(y_pred)*0.95,
             f'Mean MSE: {mean_mse:.2f}\nMean RMSE: {mean_rmse:.2f}\nMean R²: {mean_r2:.4f}',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.show()