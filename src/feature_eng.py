# src/feature_eng.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def engineer_features(path='data/processed/fused_dataset.csv'):
    df = pd.read_csv(path)

    # --- Novel engineered features ---

    # 1. NPK Ratio — a classic soil science metric
    df['npk_ratio'] = df['N'] / (df['P'] + df['K'] + 1)

    # 2. Soil fertility score — weighted combination
    df['soil_fertility_score'] = (
        0.4 * df['N'] +
        0.3 * df['P'] +
        0.3 * df['K']
    )

    # 3. Climate suitability index
    df['climate_index'] = (
        df['rainfall'] * 0.5 +
        (30 - abs(df['avg_temp'] - 25)) * 0.3 +   # optimal temp ~25C
        df['soil_humidity'] * 0.2
    )

    # 4. Soil pH deviation from neutral (7.0)
    df['ph_deviation'] = abs(df['ph'] - 7.0)

    # 5. Pesticide efficiency ratio
    df['pesticide_efficiency'] = df['yield'] / (df['pesticides'] + 1)

    # 6. Decade feature (group years into decades)
    df['decade'] = (df['Year'] // 10) * 10

    # --- Encode categoricals ---
    le_area = LabelEncoder()
    le_item = LabelEncoder()
    df['Area_enc'] = le_area.fit_transform(df['Area'])
    df['Item_enc'] = le_item.fit_transform(df['Item'])

    # Save encoders for later use in app
    import pickle
    with open('models/le_area.pkl', 'wb') as f: pickle.dump(le_area, f)
    with open('models/le_item.pkl', 'wb') as f: pickle.dump(le_item, f)

    # --- Define final feature set ---
    feature_cols = [
        'Year', 'rainfall', 'pesticides', 'avg_temp',
        'N', 'P', 'K', 'ph', 'soil_humidity',
        'npk_ratio', 'soil_fertility_score',
        'climate_index', 'ph_deviation', 'decade',
        'Area_enc', 'Item_enc'
    ]

    X = df[feature_cols]
    y = df['yield']
    
    X = X.fillna(X.median())
    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Scale ---
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    import pickle
    with open('models/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    with open('models/feature_cols.pkl', 'wb') as f: pickle.dump(feature_cols, f)

    print(f"Features: {len(feature_cols)}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train_sc, X_test_sc, y_train, y_test, X_train, X_test, df, feature_cols