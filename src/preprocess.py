# src/preprocess.py

import pandas as pd
import numpy as np

def clean_yield_data(path='data/raw/yield_df.csv'):
    df = pd.read_csv(path)

    print("YIELD DATASET")
    print(df.shape)
    print(df.isnull().sum())
    print(df.dtypes)

    df.drop_duplicates(inplace=True)

    df.rename(columns={'hg/ha_yield': 'yield',
                        'average_rain_fall_mm_per_year': 'rainfall',
                        'pesticides_tonnes': 'pesticides'}, inplace=True)

    num_cols = ['rainfall', 'pesticides', 'avg_temp', 'yield']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    Q1 = df['yield'].quantile(0.25)
    Q3 = df['yield'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['yield'] >= Q1 - 1.5*IQR) & (df['yield'] <= Q3 + 1.5*IQR)]

    df['Item'] = df['Item'].str.strip().str.title()
    df['Area'] = df['Area'].str.strip().str.title()

    print("Cleaned yield shape:", df.shape)
    df.to_csv('data/processed/cleaned_yield.csv', index=False)
    return df


def clean_soil_data(path='data/raw/soil_crop.csv'):
    df = pd.read_csv(path)

    print("\nSOIL DATASET")
    print(df.shape)
    print(df.isnull().sum())

    df.drop_duplicates(inplace=True)

    df.rename(columns={'label': 'Item'}, inplace=True)
    df['Item'] = df['Item'].str.strip().str.title()

    soil_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    for col in soil_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in ['N', 'P', 'K']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

    print("Cleaned soil shape:", df.shape)
    df.to_csv('data/processed/cleaned_soil.csv', index=False)
    return df


def fuse_datasets():
    yield_df = pd.read_csv('data/processed/cleaned_yield.csv')
    soil_df  = pd.read_csv('data/processed/cleaned_soil.csv')

    soil_agg = soil_df.groupby('Item').agg({
        'N'          : 'mean',
        'P'          : 'mean',
        'K'          : 'mean',
        'ph'         : 'mean',
        'humidity'   : 'mean',
        'temperature': 'mean',
        'rainfall'   : 'mean'
    }).reset_index()

    soil_agg.rename(columns={
        'temperature': 'soil_temp',
        'rainfall'   : 'soil_rainfall',
        'humidity'   : 'soil_humidity'
    }, inplace=True)

    fused = yield_df.merge(soil_agg, on='Item', how='left')

    matched = fused['N'].notna().sum()
    print(f"Matched {matched}/{len(fused)} rows with soil data")

    for col in ['N', 'P', 'K', 'ph', 'soil_humidity', 'soil_temp', 'soil_rainfall']:
        fused[col] = fused[col].fillna(fused[col].median())

    print("Fused dataset shape:", fused.shape)
    print("Fused columns:", list(fused.columns))

    fused.to_csv('data/processed/fused_dataset.csv', index=False)
    return fused