import pandas as pd
import joblib
from modules.features import add_temporal_features, add_geo_features
from modules.encode import encode_categorical

def load_artifacts(model_dir='models/'):
    model = joblib.load(f'{model_dir}/model.pkl')
    scaler = joblib.load(f'{model_dir}/scaler.pkl')
    product_stats = joblib.load(f'{model_dir}/product_stats.pkl')
    label_encoders = joblib.load(f'{model_dir}/label_encoders.pkl')
    feature_columns = joblib.load(f'{model_dir}/feature_columns.pkl')
    city_info = joblib.load(f'{model_dir}/city_info.pkl')
    return model, scaler, product_stats, label_encoders, feature_columns, city_info

def predict_new_data(input_df, model, scaler, product_stats, label_encoders,
                     feature_columns, city_info, threshold=0.5):
    df = input_df.copy()
    # Dates
    for col in ['ship_date','expected_delivery_date','actual_delivery_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    # Features temporelles
    df = add_temporal_features(df)
    # Features géographiques
    df = add_geo_features(df, city_info)
    # Agrégations produits
    df = df.merge(product_stats, on='product_id', how='left')
    for col in ['product_freq','product_avg_qty','product_avg_cost','product_avg_planned_delay','product_delay_rate']:
        med = product_stats[col].median()
        df[col].fillna(med, inplace=True)
    # Label encoding
    categorical_cols = list(label_encoders.keys())
    for col in categorical_cols:
        df[col+'_le'] = label_encoders[col].transform(df[col].astype(str))
    # One-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
    # Alignement
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    # Normalisation
    X = scaler.transform(df)
    proba = model.predict_proba(X)[:, 1]
    pred_binary = (proba >= threshold).astype(int)
    return proba, pred_binary