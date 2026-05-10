import pandas as pd
import numpy as np

def clean_raw_data(df):
    """Nettoie les données brutes : dates, création de la cible"""
    df = df.copy()
    date_cols = ['ship_date', 'expected_delivery_date', 'actual_delivery_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    df['is_delayed'] = (df['delivery_status'] == 'delayed').astype(int)
    df['delay_days'] = np.nan
    mask = df['delivery_status'].isin(['delivered', 'delayed'])
    df.loc[mask, 'delay_days'] = (df.loc[mask, 'actual_delivery_date'] - df.loc[mask, 'expected_delivery_date']).dt.days
    return df