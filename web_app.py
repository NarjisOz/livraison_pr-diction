from flask import Flask, request, render_template_string, send_file
import pandas as pd
import joblib
import os
import tempfile
from modules.predict import load_artifacts, predict_new_data

app = Flask(__name__)

HTML_TEMPLATE = '''
<!doctype html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <title>Prédiction des retards de livraison</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: #f4f7fc;
            margin: 0;
            padding: 40px 20px;
            color: #1e2a3e;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 24px;
            box-shadow: 0 20px 35px -10px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: all 0.2s;
        }
        .header {
            background: linear-gradient(135deg, #0b2b44, #1a4a6f);
            padding: 2rem 2rem 1.8rem;
            text-align: center;
            color: white;
        }
        .header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 600;
            letter-spacing: -0.3px;
        }
        .header p {
            margin: 0.5rem 0 0;
            opacity: 0.85;
            font-size: 0.95rem;
        }
        .content {
            padding: 2rem;
        }
        .upload-area {
            border: 2px dashed #cbd5e1;
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            background: #fafcff;
            transition: 0.2s;
            margin-bottom: 1.5rem;
        }
        .upload-area:hover {
            border-color: #2a6d9c;
            background: #f0f7ff;
        }
        input[type="file"] {
            display: none;
        }
        .file-label {
            background-color: #1e5a7d;
            color: white;
            padding: 12px 28px;
            border-radius: 40px;
            font-weight: 600;
            cursor: pointer;
            display: inline-block;
            transition: 0.2s;
            border: none;
            font-size: 1rem;
        }
        .file-label:hover {
            background-color: #0f3e58;
            transform: translateY(-1px);
        }
        .submit-btn {
            background-color: #2c7a47;
            color: white;
            border: none;
            padding: 12px 32px;
            border-radius: 40px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: 0.2s;
            margin-top: 0.5rem;
            display: inline-block;
        }
        .submit-btn:hover {
            background-color: #1f5a36;
            transform: translateY(-1px);
        }
        .info {
            background: #eef2f8;
            padding: 1rem;
            border-radius: 16px;
            font-size: 0.85rem;
            margin-top: 1.5rem;
            color: #2c3e50;
        }
        .result-box {
            background: #e9f4e9;
            border-left: 5px solid #2c7a47;
            padding: 1rem;
            border-radius: 12px;
            margin-top: 1.5rem;
        }
        .error-box {
            background: #ffe6e5;
            border-left: 5px solid #c23d3d;
            padding: 1rem;
            border-radius: 12px;
            margin-top: 1.5rem;
        }
        footer {
            text-align: center;
            font-size: 0.75rem;
            color: #6c7a89;
            border-top: 1px solid #e2e8f0;
            padding: 1.5rem;
            background: #f9fbfd;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📦 Prédiction des retards de livraison</h1>
        <p>Modèle logistique Nigeria – Classification avec seuil ajustable</p>
    </div>
    <div class="content">
        <form method="post" enctype="multipart/form-data" id="uploadForm">
            <div class="upload-area">
                <label for="fileInput" class="file-label">📂 Choisir un fichier CSV</label>
                <input type="file" name="file" id="fileInput" accept=".csv" required>
                <div style="margin-top: 1rem; font-size: 0.85rem; color: #4a627a;">Formats acceptés : CSV (encodage UTF-8, séparateur virgule)</div>
            </div>
            <div style="text-align: center;">
                <button type="submit" class="submit-btn">🚀 Lancer la prédiction</button>
            </div>
        </form>
        {% if message %}
        <div class="result-box">
            {{ message }}
        </div>
        {% endif %}
        {% if error %}
        <div class="error-box">
            ❌ {{ error }}
        </div>
        {% endif %}
        <div class="info">
            <strong>📌 Format attendu :</strong> Le fichier doit contenir les colonnes :<br>
            <code>shipment_id, product_id, supplier_name, origin_city, destination_city, ship_date, expected_delivery_date, actual_delivery_date, quantity, shipping_cost_ngn, logistics_company, delivery_status</code>
        </div>
    </div>
    <footer>
        Modèle entraîné sur données synthétiques nigérianes – Régression logistique (class_weight='balanced')
    </footer>
</div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template_string(HTML_TEMPLATE, error="Aucun fichier sélectionné")
        
        # Sauvegarde temporaire du fichier entrant
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_in:
            file.save(tmp_in.name)
            tmp_in_path = tmp_in.name
        
        try:
            # Charger les artefacts du modèle
            model, scaler, product_stats, label_encoders, feature_columns, city_info = load_artifacts('models/')
            # Lire le CSV
            df_input = pd.read_csv(tmp_in_path)
            # Prédire
            proba, pred = predict_new_data(df_input, model, scaler, product_stats,
                                           label_encoders, feature_columns, city_info,
                                           threshold=0.5)  # seuil par défaut, vous pouvez l'ajuster
            df_input['pred_prob'] = proba
            df_input['pred_delayed'] = pred
            
            # Sauvegarder le résultat dans un fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_out:
                df_input.to_csv(tmp_out.name, index=False)
                tmp_out_path = tmp_out.name
            
            # Envoyer le fichier en téléchargement
            return send_file(tmp_out_path, as_attachment=True, download_name='predictions.csv')
        
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=f"Erreur lors du traitement : {str(e)}")
        finally:
            # Nettoyer le fichier temporaire entrant
            if os.path.exists(tmp_in_path):
                os.unlink(tmp_in_path)
    
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    print("🌐 Démarrage de l'application web...")
    print("👉 Ouvrez votre navigateur et allez à l'adresse : http://127.0.0.1:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)