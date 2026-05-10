import pandas as pd
import numpy as np
from modules.utils import haversine

def add_temporal_features(df):
    df = df.copy()
    df['planned_delay'] = (df['expected_delivery_date'] - df['ship_date']).dt.days
    df['ship_month'] = df['ship_date'].dt.month
    df['ship_dayofweek'] = df['ship_date'].dt.dayofweek
    df['ship_quarter'] = df['ship_date'].dt.quarter
    df['ship_season'] = df['ship_month'].apply(
        lambda m: 0 if m in [12,1,2] else (1 if m in [3,4,5] else (2 if m in [6,7,8] else 3))
    )
    return df

def add_geo_features(df, city_info):
    df = df.copy()
    # Origine
    df = df.merge(city_info[['city','latitude','longitude','region','population']],
                  left_on='origin_city', right_on='city', how='left')
    df.rename(columns={'latitude':'origin_lat','longitude':'origin_lon',
                       'region':'origin_region','population':'origin_population'}, inplace=True)
    df.drop('city', axis=1, inplace=True)
    # Destination
    df = df.merge(city_info[['city','latitude','longitude','region','population']],
                  left_on='destination_city', right_on='city', how='left')
    df.rename(columns={'latitude':'dest_lat','longitude':'dest_lon',
                       'region':'dest_region','population':'dest_population'}, inplace=True)
    df.drop('city', axis=1, inplace=True)
    df['distance_km'] = haversine(df['origin_lat'], df['origin_lon'],
                                  df['dest_lat'], df['dest_lon'])
    return df

def add_product_aggregates(train_df, test_df):
    """Calcule les stats produits sur train et les applique à train et test"""
    product_stats = train_df.groupby('product_id').agg({
        'quantity': ['count','mean'],
        'shipping_cost_ngn': 'mean',
        'planned_delay': 'mean',
        'is_delayed': 'mean'
    }).reset_index()
    product_stats.columns = ['product_id','product_freq','product_avg_qty',
                             'product_avg_cost','product_avg_planned_delay','product_delay_rate']
    train_df = train_df.merge(product_stats, on='product_id', how='left')
    test_df = test_df.merge(product_stats, on='product_id', how='left')
    for col in ['product_freq','product_avg_qty','product_avg_cost','product_avg_planned_delay','product_delay_rate']:
        med = train_df[col].median()
        train_df[col].fillna(med, inplace=True)
        test_df[col].fillna(med, inplace=True)
    return train_df, test_df, product_stats