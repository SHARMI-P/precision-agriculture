# src/optimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pickle

def recommend_fertilizer(
    model,
    scaler,
    feature_cols,
    crop_name,
    area_name,
    current_N,
    current_P,
    current_K,
    rainfall,
    avg_temp,
    pesticides,
    ph,
    soil_humidity,
    year=2024
):
    """
    Given current soil conditions, find the optimal N, P, K
    fertilizer doses that maximize predicted crop yield.
    Uses multiple starting points to avoid local minima.
    """

    # Load encoders
    le_area = pickle.load(open('models/le_area.pkl', 'rb'))
    le_item = pickle.load(open('models/le_item.pkl', 'rb'))

    # Encode crop and area names
    try:
        area_enc = le_area.transform([area_name])[0]
    except ValueError:
        # If area not seen during training, use most common index
        area_enc = 0

    try:
        item_enc = le_item.transform([crop_name])[0]
    except ValueError:
        item_enc = 0

    decade = (year // 10) * 10

    def yield_from_npk(npk):
        N, P, K = npk

        # Prevent division by zero
        npk_ratio          = N / (P + K + 1e-6)
        soil_fertility     = 0.4 * N + 0.3 * P + 0.3 * K
        climate_index      = (
            rainfall * 0.5 +
            (30 - abs(avg_temp - 25)) * 0.3 +
            soil_humidity * 0.2
        )
        ph_deviation       = abs(ph - 7.0)
        pesticide_eff      = 0  # placeholder — yield not known yet

        row = [
            year, rainfall, pesticides, avg_temp,
            N, P, K, ph, soil_humidity,
            npk_ratio, soil_fertility,
            climate_index, ph_deviation, decade,
            area_enc, item_enc
        ]

        # Pass as DataFrame with correct column names to suppress warnings
        row_df     = pd.DataFrame([row], columns=feature_cols)
        row_scaled = scaler.transform(row_df)

        predicted  = model.predict(row_scaled)[0]

        # Return negative because minimize() minimizes (we want to maximize)
        return -predicted

    # Wider bounds for better exploration
    # N: 0-200 kg/ha, P: 0-200 kg/ha, K: 0-250 kg/ha
    bounds = [(0, 200), (0, 200), (0, 250)]

    # Multiple starting points to escape local minima
    starting_points = [
        [current_N, current_P, current_K],   # start from current values
        [20,  20,  20 ],                      # low NPK
        [50,  50,  50 ],                      # medium NPK
        [80,  60,  80 ],                      # medium-high NPK
        [100, 80,  100],                      # high NPK
        [140, 120, 180],                      # very high NPK
        [30,  90,  150],                      # P and K heavy
        [120, 40,  60 ],                      # N heavy
    ]

    best_result = None

    for start in starting_points:
        try:
            result = minimize(
                yield_from_npk,
                x0      = start,
                method  = 'L-BFGS-B',
                bounds  = bounds,
                options = {
                    'maxiter' : 1000,
                    'ftol'    : 1e-12,
                    'gtol'    : 1e-8
                }
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        except Exception:
            continue

    # Extract optimal values
    opt_N, opt_P, opt_K = best_result.x
    max_yield            = -best_result.fun

    # Also compute baseline yield at current NPK for comparison
    baseline_row = [
        year, rainfall, pesticides, avg_temp,
        current_N, current_P, current_K, ph, soil_humidity,
        current_N / (current_P + current_K + 1e-6),
        0.4 * current_N + 0.3 * current_P + 0.3 * current_K,
        rainfall * 0.5 + (30 - abs(avg_temp - 25)) * 0.3 + soil_humidity * 0.2,
        abs(ph - 7.0),
        decade,
        area_enc, item_enc
    ]
    baseline_df    = pd.DataFrame([baseline_row], columns=feature_cols)
    baseline_scaled= scaler.transform(baseline_df)
    baseline_yield = model.predict(baseline_scaled)[0]

    improvement    = max_yield - baseline_yield
    improvement_pct= (improvement / baseline_yield * 100) if baseline_yield > 0 else 0

    # Print full report to terminal
    print(f"\n{'='*52}")
    print(f"  Fertilizer Recommendation — {crop_name}")
    print(f"{'='*52}")
    print(f"  Region         : {area_name}")
    print(f"  Year           : {year}")
    print(f"  Rainfall       : {rainfall} mm/year")
    print(f"  Avg Temp       : {avg_temp} °C")
    print(f"  Soil pH        : {ph}")
    print(f"  Soil Humidity  : {soil_humidity}%")
    print(f"{'-'*52}")
    print(f"  Current  N={current_N:.1f}  P={current_P:.1f}  K={current_K:.1f}")
    print(f"  Optimal  N={opt_N:.1f}  P={opt_P:.1f}  K={opt_K:.1f}")
    print(f"{'-'*52}")
    print(f"  Baseline Yield : {baseline_yield:,.0f} hg/ha")
    print(f"  Optimal Yield  : {max_yield:,.0f} hg/ha")
    print(f"  Improvement    : +{improvement:,.0f} hg/ha  ({improvement_pct:.1f}%)")
    print(f"{'-'*52}")
    print(f"  Change N       : {opt_N - current_N:+.1f} kg/ha")
    print(f"  Change P       : {opt_P - current_P:+.1f} kg/ha")
    print(f"  Change K       : {opt_K - current_K:+.1f} kg/ha")
    print(f"{'='*52}\n")

    return {
        'crop'            : crop_name,
        'area'            : area_name,
        'current_N'       : round(current_N, 2),
        'current_P'       : round(current_P, 2),
        'current_K'       : round(current_K, 2),
        'optimal_N'       : round(opt_N,      2),
        'optimal_P'       : round(opt_P,      2),
        'optimal_K'       : round(opt_K,      2),
        'delta_N'         : round(opt_N - current_N, 2),
        'delta_P'         : round(opt_P - current_P, 2),
        'delta_K'         : round(opt_K - current_K, 2),
        'baseline_yield'  : round(baseline_yield, 2),
        'predicted_yield' : round(max_yield,      2),
        'improvement_hgha': round(improvement,    2),
        'improvement_pct' : round(improvement_pct,2),
    }


def batch_recommend(model, scaler, feature_cols, crops_list):
    """
    Run optimizer for multiple crops at once.
    crops_list is a list of dicts, each with keys:
    crop_name, area_name, N, P, K, rainfall, avg_temp,
    pesticides, ph, soil_humidity, year
    """
    results = []
    for crop_config in crops_list:
        result = recommend_fertilizer(
            model        = model,
            scaler       = scaler,
            feature_cols = feature_cols,
            crop_name    = crop_config['crop_name'],
            area_name    = crop_config['area_name'],
            current_N    = crop_config.get('N',           50),
            current_P    = crop_config.get('P',           50),
            current_K    = crop_config.get('K',           50),
            rainfall     = crop_config.get('rainfall',  1000),
            avg_temp     = crop_config.get('avg_temp',    25),
            pesticides   = crop_config.get('pesticides',  10),
            ph           = crop_config.get('ph',         6.5),
            soil_humidity= crop_config.get('soil_humidity',70),
            year         = crop_config.get('year',      2024),
        )
        results.append(result)

    print(f"\nBatch complete — {len(results)} crops optimized")
    return results