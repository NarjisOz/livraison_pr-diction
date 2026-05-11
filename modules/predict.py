import pandas as pd
import numpy as np
import joblib
import json
from modules.features import add_temporal_features, add_geo_features

def load_artifacts(model_dir='models/'):
    model = joblib.load(f'{model_dir}/model.pkl')
    scaler = joblib.load(f'{model_dir}/scaler.pkl')
    product_stats = joblib.load(f'{model_dir}/product_stats.pkl')
    label_encoders = joblib.load(f'{model_dir}/label_encoders.pkl')
    feature_columns = joblib.load(f'{model_dir}/feature_columns.pkl')
    city_info = joblib.load(f'{model_dir}/city_info.pkl')
    with open(f'{model_dir}/governance_report.json', 'r') as f:
        governance_report = json.load(f)
    return model, scaler, product_stats, label_encoders, feature_columns, city_info, governance_report

def predict_new_data(input_df, model, scaler, product_stats, label_encoders,
                     feature_columns, city_info, governance_report, threshold=0.5):
    df = input_df.copy()
    # Dates
    for col in ['ship_date', 'expected_delivery_date', 'actual_delivery_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            
    # Temporal
    df = add_temporal_features(df)
    
    # Geo (only if governance report says to use it)
    if governance_report.get('use_geo', False):
        df = add_geo_features(df, city_info)
        
    # Product stats
    df = df.merge(product_stats, on='product_id', how='left')
    for col in ['product_freq', 'product_avg_qty', 'product_avg_cost', 'product_avg_planned_delay', 'product_delay_rate']:
        med = product_stats[col].median()
        df[col] = df[col].fillna(med)
        
    # Encoding
    cat_cols = ['supplier_name', 'origin_city', 'destination_city', 'logistics_company']
    if 'origin_region' in df.columns:
        cat_cols.extend(['origin_region', 'dest_region'])
        
    for col in cat_cols:
        if col in label_encoders and col in df.columns:
            df[col+'_le'] = label_encoders[col].transform(df[col].astype(str))
            
    df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, drop_first=True)
    
    # Align
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    
    # Fill remaining NaNs
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64', 'int32']:
            df[col] = df[col].fillna(df[col].median())
            
    # Scale and predict
    X = scaler.transform(df)
    proba = model.predict_proba(X)[:, 1]
    pred_binary = (proba >= threshold).astype(int)
    return proba, pred_binary, X, df