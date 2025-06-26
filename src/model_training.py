import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

def prepare_dataset(df):
    df = df.copy()
    df.dropna(inplace=True)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    X = df.drop(['close', 'returns', 'log_returns', 'target'], axis=1, errors='ignore')
    y = df['target']
    return X, y

def train_xgb_tuned(X, y):
    """
    Train an XGBClassifier with randomized hyperparameter search.
    """
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=20,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    search.fit(X, y)
    print("✅ Best params:", search.best_params_)
    print("✅ Best CV accuracy:", search.best_score_)
    return search.best_estimator_

def predict_proba_direction(model, X, threshold=0.6):
    """
    Generate binary up/down signals based on probability threshold.
    """
    proba = model.predict_proba(X)[:, 1]
    return (proba > threshold).astype(int)

def save_model(model, filename="models/model.pkl"):
    joblib.dump(model, filename)
    print(f"\n📦 Model saved to: {filename}")

def load_model(filename="models/model.pkl"):
    return joblib.load(filename)
