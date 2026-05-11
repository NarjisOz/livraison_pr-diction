import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from modules.predict import load_artifacts, predict_new_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_cli.py input.csv [output.csv]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else '../data/predictions/predictions.csv'
    print("Chargement des artefacts...")
    model, scaler, product_stats, label_encoders, feature_columns, city_info = load_artifacts('models/')
    print(f"Chargement de {input_file}...")
    new_data = pd.read_csv(input_file)
    print("Prédiction en cours...")
    proba, pred = predict_new_data(new_data, model, scaler, product_stats,
                                   label_encoders, feature_columns, city_info)
    new_data['pred_prob'] = proba
    new_data['pred_delayed'] = pred
    new_data.to_csv(output_file, index=False)
    print(f"✅ Prédictions sauvegardées dans {output_file}")

if __name__ == "__main__":
    main()