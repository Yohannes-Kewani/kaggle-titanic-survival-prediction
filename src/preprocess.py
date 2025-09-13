import pandas as pd

def load_data(path):
    """Load dataset from CSV"""
    return pd.read_csv(path)

def clean_data(df, is_train=True):
    """
    Basic cleaning: handle missing values, remove features with small effect,
    and encode categorical values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame (train or test).
    is_train : bool
        If True, drop PassengerId and target leakage columns (train.csv).
        If False, keep PassengerId for submission (test.csv).
    """
    # Drop unused columns
    drop_cols = ['Name', 'Ticket', 'Cabin']
    if is_train:
        drop_cols.append('PassengerId')  # keep PassengerId in test set
    
    df = df.drop(columns=drop_cols, axis=1, errors="ignore")

    # Remove duplicates (extra safety)
    df = df.drop_duplicates()

    # Fill missing Age with median
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())

    # Fill missing Fare (appears in test set sometimes)
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Fill missing Embarked with mode
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Encode categorical features
    df = pd.get_dummies(df, drop_first=True)
    # Change true, false values to numerical
    df = df.astype(int)

    return df
