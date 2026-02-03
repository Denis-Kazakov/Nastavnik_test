from pickle import dump

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

def learning_curve(series: pd.Series):
    """Helper function to calculate the mean success rate over the last 3 attempts"""
    return series.tail(3).mean()

def load_interactions(filepath: str) -> pd.DataFrame:
    """
    Load training data
    input: JSON
    output: Pandas Dataframe with columns [user_id, skill_id, correctness, attempt_num]
    """
    df = pd.read_json(filepath)
    print(f'Количество уникальных студентов: {df.user_id.nunique()}')
    print(f'Количество уникальных навыков: {df.skill_id.nunique()}')
    print(f'Средний процент правильных ответов по всем попыткам: {df.correctness.mean().item()}')
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create training features"""
    features_df = df.groupby(['user_id', 'skill_id'], as_index=False)\
        .agg({'attempt_num': 'count', 'correctness': ['mean', 'last', learning_curve],})
    features_df.columns = [' '.join(col).strip() for col in features_df.columns.values]
    features_df.rename(
        columns={
            'attempt_num count': 'num_attempts',
            'correctness mean': 'success_rate',
            'correctness last': 'last_correct',
            'correctness learning_curve': 'learning_curve'
        },
        inplace=True
    )
    return features_df

def train_model(features_df: pd.DataFrame) -> tuple:
    """Train a logistic regression model"""
    features_df['target'] = features_df.success_rate > 0.75
    X = features_df.drop(columns=['user_id', 'skill_id', 'target'])
    y = features_df['target']
    scaler = MinMaxScaler()
    X['num_attempts'] = scaler.fit_transform(X[['num_attempts']])
    clf = LogisticRegression().fit(X, y)
    print(f'Точность модели на обучающих данных: clf.score(X, y)')
    return clf, scaler

DATA_SOURCE = 'interactions.json'
df = load_interactions(DATA_SOURCE)
feature_df = engineer_features(df)
clf, scaler = train_model(feature_df)

with open("model.pkl", "wb") as f:
    dump(clf, f, protocol=5)

with open("scaler.pkl", "wb") as f:
    dump(scaler, f, protocol=5)
