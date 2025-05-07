import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_data(df, categorical_cols, numerical_cols):
    """
    Simple preprocessing: one-hot encode categoricals, scale numericals.
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    scaler = StandardScaler()

    X_cat = encoder.fit_transform(df[categorical_cols])
    X_num = scaler.fit_transform(df[numerical_cols])

    import numpy as np
    X_all = np.hstack((X_cat, X_num))
    return X_all, encoder, scaler
