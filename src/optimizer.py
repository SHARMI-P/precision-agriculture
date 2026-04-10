# src/optimizer.py

import numpy as np
from scipy.optimize import minimize
import pickle

def recommend_fertilizer(model, scaler, feature_cols,
                          crop_name, area_name,
                          current_N, current_P, current_K,
                          rainfall, avg_temp, pesticides,
                          ph, soil_humidity, year=2024):
    """
    Given current soil conditions, find the optimal N, P, K
    fertilizer doses that maximize predicted yield.
    """

    le_area = pickle.load(open('models/le_area.pkl','rb'))
    le_item = pickle.load(open('models/le_item.pkl','rb'))

    area_enc = le_area.transform([area_name])[0]
    item_enc = le_item.transform([crop_name])[0]

    def yield_from_npk(npk):
        N, P, K = npk
        npk_ratio       = N / (P + K + 1)
        soil_fertility  = 0.4*N + 0.3*P + 0.3*K
        climate_index   = rainfall*0.5 + (30 - abs(avg_temp - 25))*0.3 + soil_humidity*0.2
        ph_deviation    = abs(ph - 7.0)
        decade          = (year // 10) * 10

        row = [year, rainfall, pesticides, avg_temp,
               N, P, K, ph, soil_humidity,
               npk_ratio, soil_fertility,
               climate_index, ph_deviation, decade,
               area_enc, item_enc]

        row_scaled = scaler.transform([row])
        predicted_yield = model.predict(row_scaled)[0]
        return -predicted_yield  # negative because minimize() minimizes

    # Bounds: N in [0,140], P in [0,145], K in [0,205] (kg/ha typical range)
    bounds = [(0, 140), (0, 145), (0, 205)]
    x0 = [current_N, current_P, current_K]

    result = minimize(yield_from_npk, x0, method='L-BFGS-B', bounds=bounds)

    opt_N, opt_P, opt_K = result.x
    max_yield = -result.fun

    print(f"\n=== Fertilizer Recommendation for {crop_name} ===")
    print(f"Current  : N={current_N:.1f}, P={current_P:.1f}, K={current_K:.1f}")
    print(f"Optimal  : N={opt_N:.1f},  P={opt_P:.1f},  K={opt_K:.1f}")
    print(f"Predicted Max Yield: {max_yield:.0f} hg/ha")
    print(f"Change   : N {opt_N-current_N:+.1f}, P {opt_P-current_P:+.1f}, K {opt_K-current_K:+.1f}")

    return {
        'optimal_N': round(opt_N, 2),
        'optimal_P': round(opt_P, 2),
        'optimal_K': round(opt_K, 2),
        'predicted_yield': round(max_yield, 2),
        'delta_N': round(opt_N - current_N, 2),
        'delta_P': round(opt_P - current_P, 2),
        'delta_K': round(opt_K - current_K, 2)
    }