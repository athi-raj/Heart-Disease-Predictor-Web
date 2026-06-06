from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, os

from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='static')
CORS(app)

# ── Load artifacts ──
model           = load_model('models/heart_disease_model.keras')
scaler          = joblib.load('models/heart_scaler.pkl')
feature_columns = joblib.load('models/heart_features.pkl')

explainer = shap.Explainer(
    lambda x: model.predict(x, verbose=0),
    np.zeros((1, len(feature_columns)))
)

feature_mapping = {
    "age"                     : "Patient Age",
    "trestbps"                : "Resting Blood Pressure",
    "chol"                    : "Cholesterol Level",
    "thalch"                  : "Maximum Heart Rate",
    "oldpeak"                 : "ST Depression",
    "ca"                      : "Major Vessels Count",
    "fbs_True"                : "High Fasting Blood Sugar",
    "fbs_False"               : "Normal Fasting Blood Sugar",
    "cp_typical angina"       : "Typical Angina Chest Pain",
    "cp_atypical angina"      : "Atypical Angina Chest Pain",
    "cp_non-anginal"          : "Non-Anginal Chest Pain",
    "cp_asymptomatic"         : "Asymptomatic Chest Pain",
    "slope_upsloping"         : "Upsloping ST Segment",
    "slope_flat"              : "Flat ST Segment",
    "slope_downsloping"       : "Downsloping ST Segment",
    "sex_Female"              : "Female Gender",
    "sex_Male"                : "Male Gender",
    "restecg_normal"          : "Normal ECG",
    "restecg_st-t abnormality": "ST-T Wave Abnormality ECG",
    "restecg_lv hypertrophy"  : "Left Ventricular Hypertrophy ECG",
    "thal_normal"             : "Normal Thalassemia",
    "thal_fixed defect"       : "Fixed Thalassemia Defect",
    "thal_reversable defect"  : "Reversible Thalassemia Defect",
    "exang_True"              : "Exercise Induced Angina",
    "exang_False"             : "No Exercise Induced Angina",
    "dataset_Cleveland"       : "Cleveland Dataset",
    "dataset_Hungary"         : "Hungary Dataset",
    "dataset_Switzerland"     : "Switzerland Dataset",
    "dataset_VA Long Beach"   : "VA Long Beach Dataset",
}

recommendations = {
    "Exercise Induced Angina"      : "Avoid strenuous physical activity. Consult a cardiologist for a stress test. Monitor active heart rate regularly.",
    "Asymptomatic Chest Pain"      : "Silent chest pain is a serious indicator. Schedule an immediate cardiac evaluation.",
    "Typical Angina Chest Pain"    : "Typical angina requires medical review. Avoid triggers like cold weather and heavy meals.",
    "High Fasting Blood Sugar"     : "Manage blood sugar through diet and medication. Consult an endocrinologist.",
    "Cholesterol Level"            : "Reduce saturated fats and increase fibre intake. Schedule a lipid profile test.",
    "ST Depression"                : "ST depression indicates reduced blood flow. Seek immediate cardiac evaluation.",
    "Downsloping ST Segment"       : "Downsloping ST segment is a high-risk ECG pattern. Consult a cardiologist urgently.",
    "Flat ST Segment"              : "Flat ST segment may indicate ischaemia. Schedule an ECG review.",
    "Fixed Thalassemia Defect"     : "Fixed thalassemia defect indicates permanent heart damage. Regular cardiac monitoring required.",
    "Reversible Thalassemia Defect": "Reversible defect suggests reduced blood flow. Consult for nuclear stress test.",
    "Major Vessels Count"          : "Blocked vessels significantly increase risk. Angiography evaluation recommended.",
    "Resting Blood Pressure"       : "High blood pressure strains the heart. Reduce salt intake and monitor daily.",
    "Patient Age"                  : "Age is a non-modifiable risk factor. Increase frequency of cardiac checkups.",
    "Left Ventricular Hypertrophy ECG": "LV hypertrophy indicates heart strain. Blood pressure management is critical.",
}

def get_recommendation(feature_name):
    for key in recommendations:
        if key.lower() in feature_name.lower():
            return recommendations[key]
    return "Maintain a healthy lifestyle. Regular exercise, balanced diet, and routine checkups are recommended."


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    patient = pd.DataFrame([{
        'age'     : float(data['age']),
        'sex'     : data['sex'],
        'dataset' : data['dataset'],
        'cp'      : data['cp'],
        'trestbps': float(data['trestbps']),
        'chol'    : float(data['chol']),
        'fbs'     : data['fbs'],
        'restecg' : data['restecg'],
        'thalch'  : float(data['thalch']),
        'exang'   : data['exang'],
        'oldpeak' : float(data['oldpeak']),
        'slope'   : data['slope'],
        'ca'      : float(data['ca']),
        'thal'    : data['thal'],
    }])

    patient_encoded = pd.get_dummies(patient)
    patient_encoded = patient_encoded.reindex(columns=feature_columns, fill_value=0)
    patient_scaled  = scaler.transform(patient_encoded)

    probability = float(model.predict(patient_scaled, verbose=0)[0][0])

    if probability < 0.40:
        risk_level  = 'LOW'
        risk_class  = 'low'
        result_text = '✓ LOW RISK OF HEART DISEASE'
    elif probability < 0.60:
        risk_level  = 'MODERATE'
        risk_class  = 'moderate'
        result_text = '⚠ MODERATE RISK OF HEART DISEASE'
    else:
        risk_level  = 'VERY HIGH'
        risk_class  = 'high'
        result_text = '🔴 VERY HIGH RISK OF HEART DISEASE'

    confidence = max(probability, 1 - probability) * 100

    # ── SHAP ──
    shap_values = explainer(patient_scaled)
    vals        = shap_values.values[0].ravel()

    top_idx     = np.argsort(np.abs(vals))[-10:]
    top_vals    = vals[top_idx]
    top_names   = [feature_mapping.get(feature_columns[i], feature_columns[i]) for i in top_idx]
    colors      = ['#f85149' if v > 0 else '#3fb950' for v in top_vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top_names, top_vals, color=colors)
    ax.axvline(0, color='white', linewidth=0.8)
    ax.set_title('Top 10 Features Influencing This Prediction', color='#e6edf3', fontsize=12, pad=10)
    ax.set_xlabel('SHAP Value  (🔴 increases risk · 🟢 reduces risk)', color='#8a9bb0', fontsize=9)
    ax.tick_params(colors='#8a9bb0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#21262d')
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    shap_chart = base64.b64encode(buf.read()).decode('utf-8')

    top_risk_idx    = int(np.argmax(top_vals)) if top_vals.max() > 0 else None
    top_prot_idx    = int(np.argmin(top_vals)) if top_vals.min() < 0 else None
    top_risk_name   = top_names[top_risk_idx]   if top_risk_idx    is not None else 'N/A'
    top_prot_name   = top_names[top_prot_idx]   if top_prot_idx    is not None else 'N/A'
    recommendation  = get_recommendation(top_risk_name)

    return jsonify({
        'probability'    : round(probability, 4),
        'confidence'     : round(confidence, 2),
        'risk_level'     : risk_level,
        'risk_class'     : risk_class,
        'result_text'    : result_text,
        'shap_chart'     : shap_chart,
        'top_risk'       : top_risk_name,
        'top_protect'    : top_prot_name,
        'recommendation' : recommendation,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
