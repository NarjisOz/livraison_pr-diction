import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical(train_df, test_df):
    categorical_cols = ['supplier_name','origin_city','destination_city',
                        'logistics_company','origin_region','dest_region']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col+'_le'] = le.fit_transform(train_df[col].astype(str))
        test_df[col+'_le'] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le
    # One-hot encoding
    train_df = pd.get_dummies(train_df, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
    # Alignement des colonnes
    missing_cols = set(train_df.columns) - set(test_df.columns)
    for col in missing_cols:
        test_df[col] = 0
    test_df = test_df[train_df.columns]
    feature_columns = train_df.columns
    return train_df, test_df, label_encoders, feature_columns