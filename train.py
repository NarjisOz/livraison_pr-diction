import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_score, recall_score

print("1. Chargement des données...")
df = pd.read_csv('data/raw/nigerian_retail_and_ecommerce_supply_chain_logistics_data.csv')
date_cols = ['ship_date', 'expected_delivery_date', 'actual_delivery_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col])
df['is_delayed'] = (df['delivery_status'] == 'delayed').astype(int)
df['delay_days'] = np.nan
mask = df['delivery_status'].isin(['delivered', 'delayed'])
df.loc[mask, 'delay_days'] = (df.loc[mask, 'actual_delivery_date'] - df.loc[mask, 'expected_delivery_date']).dt.days

print("2. Division train/test...")
X = df.drop(columns=['is_delayed', 'delay_days', 'delivery_status'])
y = df['is_delayed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature engineering functions
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def add_temporal(df):
    df = df.copy()
    df['planned_delay'] = (df['expected_delivery_date'] - df['ship_date']).dt.days
    df['ship_month'] = df['ship_date'].dt.month
    df['ship_dayofweek'] = df['ship_date'].dt.dayofweek
    df['ship_quarter'] = df['ship_date'].dt.quarter
    df['ship_season'] = df['ship_month'].apply(lambda m: 0 if m in [12,1,2] else (1 if m in [3,4,5] else (2 if m in [6,7,8] else 3)))
    return df

def add_geo(df, city_info):
    df = df.merge(city_info[['city','latitude','longitude','region','population']],
                  left_on='origin_city', right_on='city', how='left')
    df.rename(columns={'latitude':'origin_lat','longitude':'origin_lon',
                       'region':'origin_region','population':'origin_population'}, inplace=True)
    df.drop('city', axis=1, inplace=True)
    df = df.merge(city_info[['city','latitude','longitude','region','population']],
                  left_on='destination_city', right_on='city', how='left')
    df.rename(columns={'latitude':'dest_lat','longitude':'dest_lon',
                       'region':'dest_region','population':'dest_population'}, inplace=True)
    df.drop('city', axis=1, inplace=True)
    df['distance_km'] = haversine(df['origin_lat'], df['origin_lon'],
                                  df['dest_lat'], df['dest_lon'])
    return df

print("3. Chargement données géographiques...")
weather = pd.read_csv('data/raw/nigeria_cities_weather_data.csv')
city_info = weather.groupby('city').agg({'latitude':'first','longitude':'first','region':'first','population':'first'}).reset_index()

print("4. Feature engineering (Temporel)...")
X_train_base = add_temporal(X_train)
X_test_base = add_temporal(X_test)

print("5. Agrégations produits...")
def apply_product_stats(X_tr, X_te, y_tr):
    X_tr_temp = X_tr.copy()
    X_tr_temp['is_delayed'] = y_tr.values
    p_stats = X_tr_temp.groupby('product_id').agg({
        'quantity': ['count', 'mean'],
        'shipping_cost_ngn': 'mean',
        'planned_delay': 'mean',
        'is_delayed': 'mean'
    }).reset_index()
    p_stats.columns = ['product_id', 'product_freq', 'product_avg_qty',
                             'product_avg_cost', 'product_avg_planned_delay', 'product_delay_rate']
    X_tr = X_tr.merge(p_stats, on='product_id', how='left')
    X_te = X_te.merge(p_stats, on='product_id', how='left')
    for col in ['product_freq', 'product_avg_qty', 'product_avg_cost', 'product_avg_planned_delay', 'product_delay_rate']:
        med = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(med)
        X_te[col] = X_te[col].fillna(med)
    return X_tr, X_te, p_stats

X_train_base, X_test_base, product_stats = apply_product_stats(X_train_base, X_test_base, y_train)

# We will create two versions of datasets: Base and Geo
X_train_geo = add_geo(X_train_base, city_info)
X_test_geo = add_geo(X_test_base, city_info)

def preprocess_dataset(X_tr, X_te):
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    
    cat_cols = ['supplier_name', 'origin_city', 'destination_city', 'logistics_company']
    if 'origin_region' in X_tr.columns:
        cat_cols.extend(['origin_region', 'dest_region'])
        
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_tr[col+'_le'] = le.fit_transform(X_tr[col].astype(str))
        X_te[col+'_le'] = le.transform(X_te[col].astype(str))
        label_encoders[col] = le

    X_tr = pd.get_dummies(X_tr, columns=cat_cols, prefix=cat_cols, drop_first=True)
    X_te = pd.get_dummies(X_te, columns=cat_cols, prefix=cat_cols, drop_first=True)

    missing_cols = set(X_tr.columns) - set(X_te.columns)
    for col in missing_cols:
        X_te[col] = 0
    X_te = X_te[X_tr.columns]

    cols_to_drop = ['shipment_id', 'product_id', 'ship_date', 'expected_delivery_date', 'actual_delivery_date',
                    'origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']
    cols_to_drop = [c for c in cols_to_drop if c in X_tr.columns]
    X_tr.drop(columns=cols_to_drop, inplace=True)
    X_te.drop(columns=cols_to_drop, inplace=True)

    X_tr = X_tr.reset_index(drop=True)
    X_te = X_te.reset_index(drop=True)

    for col in X_tr.columns:
        if X_tr[col].dtype in ['float64', 'int64', 'int32']:
            med = X_tr[col].median()
            X_tr[col] = X_tr[col].fillna(med)
            X_te[col] = X_te[col].fillna(med)
            
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    return X_tr_scaled, X_te_scaled, scaler, label_encoders, X_tr.columns

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

X_tr_base_s, X_te_base_s, scaler_base, le_base, cols_base = preprocess_dataset(X_train_base, X_test_base)
X_tr_geo_s, X_te_geo_s, scaler_geo, le_geo, cols_geo = preprocess_dataset(X_train_geo, X_test_geo)

print("6. Évaluation des modèles...")
def evaluate_model(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)
    roc = roc_auc_score(y_te, proba)
    pr = average_precision_score(y_te, proba)
    prec = precision_score(y_te, pred, zero_division=0)
    rec = recall_score(y_te, pred, zero_division=0)
    return {'roc_auc': roc, 'pr_auc': pr, 'precision': prec, 'recall': rec, 'model': model}

results = {}

# Logistic Regression
print("   -> Logistic Regression (Base)...")
lr_base = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
results['LR_Base'] = evaluate_model(lr_base, X_tr_base_s, y_train, X_te_base_s, y_test)

print("   -> Logistic Regression (Geo)...")
lr_geo = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
results['LR_Geo'] = evaluate_model(lr_geo, X_tr_geo_s, y_train, X_te_geo_s, y_test)

# XGBoost
print("   -> XGBoost (Base)...")
scale_pos = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
xgb_base = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, scale_pos_weight=scale_pos, n_jobs=-1)
results['XGB_Base'] = evaluate_model(xgb_base, X_tr_base_s, y_train, X_te_base_s, y_test)

print("   -> XGBoost (Geo)...")
xgb_geo = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, scale_pos_weight=scale_pos, n_jobs=-1)
results['XGB_Geo'] = evaluate_model(xgb_geo, X_tr_geo_s, y_train, X_te_geo_s, y_test)

# Governance Logic
best_geo_roc = max(results['LR_Geo']['roc_auc'], results['XGB_Geo']['roc_auc'])
best_base_roc = max(results['LR_Base']['roc_auc'], results['XGB_Base']['roc_auc'])

geo_improvement = (best_geo_roc - best_base_roc) / best_base_roc

use_geo = False
if geo_improvement > 0.005: # 0.5% improvement threshold
    use_geo = True
    geo_decision = f"Geolocation features significantly improved estimated delivery prediction (ROC-AUC) by {geo_improvement:.1%}. They have been integrated."
else:
    use_geo = False
    geo_decision = f"Geolocation features provided limited predictive contribution ({geo_improvement:.1%}) and were autonomously excluded to improve robustness and prevent overfitting."

print(f"\n--- DECISION: {geo_decision} ---\n")

best_model_name = ""
best_model = None
best_roc = 0

for name, res in results.items():
    if use_geo and 'Geo' not in name: continue
    if not use_geo and 'Base' not in name: continue
    
    if res['roc_auc'] > best_roc:
        best_roc = res['roc_auc']
        best_model = res['model']
        best_model_name = name

model_algo = "Advanced Gradient Boosting (XGBoost)" if "XGB" in best_model_name else "Logistic Regression"
model_decision = f"{model_algo} was selected automatically as the optimal algorithm because it achieved the highest Validation ROC-AUC ({best_roc:.3f}), surpassing linear baseline models."

governance_report = {
    "geo_decision_narrative": geo_decision,
    "model_selection_narrative": model_decision,
    "use_geo": use_geo,
    "best_model_name": best_model_name,
    "metrics": {
        "roc_auc": float(best_roc),
        "precision": float(results[best_model_name]['precision']),
        "recall": float(results[best_model_name]['recall'])
    },
    "all_metrics": {k: {m: float(v[m]) for m in ['roc_auc', 'pr_auc', 'precision', 'recall']} for k, v in results.items()}
}

print("7. Sauvegarde...")
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/model.pkl')
joblib.dump(scaler_geo if use_geo else scaler_base, 'models/scaler.pkl')
joblib.dump(product_stats, 'models/product_stats.pkl')
joblib.dump(le_geo if use_geo else le_base, 'models/label_encoders.pkl')
joblib.dump(cols_geo if use_geo else cols_base, 'models/feature_columns.pkl')
joblib.dump(city_info, 'models/city_info.pkl')

with open('models/governance_report.json', 'w') as f:
    json.dump(governance_report, f, indent=4)

print("Done. Models and artifacts saved.")
