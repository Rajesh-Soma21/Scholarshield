"""
app.py — ScholarShield Flask Backend
=====================================
Serves the web app and runs all 4 ML models on each prediction request.
"""

import os, json, joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, 'models')

# ── Load all 4 models at startup ──────────────────────────────────────────────
MODELS = {}
MODEL_KEYS = ['naive_bayes', 'logistic_regression', 'decision_tree', 'svm']
DISPLAY_NAMES = {
    'naive_bayes':          'Naive Bayes',
    'logistic_regression':  'Logistic Regression',
    'decision_tree':        'Decision Tree',
    'svm':                  'SVM (LinearSVC)',
}
ICONS = {
    'naive_bayes':         '📊',
    'logistic_regression': '📈',
    'decision_tree':       '🌳',
    'svm':                 '⚡',
}

for key in MODEL_KEYS:
    path = os.path.join(MODELS_DIR, f'{key}.pkl')
    if os.path.exists(path):
        MODELS[key] = joblib.load(path)
        print(f"  ✅ Loaded: {DISPLAY_NAMES[key]}")
    else:
        print(f"  ⚠️  Missing: {path} — run train_models.py first")

# Load training metadata (accuracy, F1, etc.)
META = {}
meta_path = os.path.join(MODELS_DIR, 'meta.json')
if os.path.exists(meta_path):
    with open(meta_path) as f:
        META = json.load(f)


def build_feature(data: dict) -> str:
    """Combine all input fields into a single text feature string."""
    parts = [
        data.get('scholarship_name', ''),
        data.get('provider', ''),
        data.get('description', ''),
        data.get('url', ''),
        data.get('contact_email', ''),
        data.get('application_fee', ''),
        data.get('amount', ''),
    ]
    return ' '.join(p.strip() for p in parts if p and p.strip())


def predict_all(feature_text: str) -> dict:
    """Run all 4 models and compute ensemble vote."""
    predictions = {}
    votes = {'REAL': 0, 'FAKE': 0}

    for key, model in MODELS.items():
        pred = model.predict([feature_text])[0]
        label = 'FAKE' if pred == 1 else 'REAL'
        votes[label] += 1

        # Confidence score
        try:
            if hasattr(model.named_steps['clf'], 'predict_proba'):
                proba = model.predict_proba([feature_text])[0]
                conf = float(max(proba)) * 100
            elif hasattr(model.named_steps['clf'], 'decision_function'):
                raw = model.decision_function([feature_text])[0]
                conf = min(99.9, max(50.1, 50 + abs(float(raw)) * 18))
            else:
                conf = 85.0
        except Exception:
            conf = 80.0

        model_meta = META.get('results', {}).get(DISPLAY_NAMES[key], {})
        predictions[key] = {
            'display_name': DISPLAY_NAMES[key],
            'icon':         ICONS[key],
            'label':        label,
            'confidence':   round(conf, 1),
            'train_acc':    model_meta.get('accuracy', 0),
            'train_f1':     model_meta.get('f1_score', 0),
            'cv_mean':      model_meta.get('cv_mean', 0),
        }

    # Ensemble verdict (majority vote, with tie → SUSPICIOUS)
    total = len(MODELS)
    fake_votes = votes['FAKE']
    real_votes = votes['REAL']

    if fake_votes > real_votes:
        ensemble_label = 'FAKE'
    elif real_votes > fake_votes:
        ensemble_label = 'REAL'
    else:
        ensemble_label = 'SUSPICIOUS'

    ensemble_conf = round(max(fake_votes, real_votes) / total * 100, 1)

    # Risk factor analysis
    text_lower = feature_text.lower()
    risk_flags = []
    safe_flags = []

    suspicious_kws = ['urgent','guaranteed','100%','wire transfer','western union','moneygram',
                      'act now','limited time','secret','instant approval','processing fee',
                      'security deposit','advance fee','pay now','send money','transfer fee',
                      'otp','aadhaar number','bank account number','whatsapp','hotmail','yahoo.com',
                      'lottery','won','selected randomly','no essay','no criteria','no documents',
                      'click link','expires tonight','expires soon','24 hours','48 hours',
                      'moneyback','refundable','100 percent']

    official_kws = ['.gov.in','.edu','.ac.in','ministry','ugc','aicte','nic.in','gov.in',
                    'merit based','income criteria','official portal','no fee','no application fee',
                    'transparent','direct bank','neft','no middlemen','official website']

    for kw in suspicious_kws:
        if kw in text_lower:
            risk_flags.append(kw)
    for kw in official_kws:
        if kw in text_lower:
            safe_flags.append(kw)

    has_fee = any(c.isdigit() for c in feature_text) and any(
        w in text_lower for w in ['fee','charge','payment','pay','deposit','rupees','dollars'])
    has_gmail_yahoo = any(d in text_lower for d in ['gmail.com','yahoo.com','hotmail.com','outlook.com'])
    has_official_email = any(d in text_lower for d in ['.gov.in','.edu','.ac.in','.org','nic.in'])
    has_official_url = any(d in text_lower for d in ['.gov.in','.edu','.ac.in','ugc.','aicte.'])

    flags = []
    if has_fee and ensemble_label != 'REAL':
        flags.append({'type': 'danger', 'text': 'Application fee / payment required — major red flag'})
    if has_gmail_yahoo:
        flags.append({'type': 'danger', 'text': 'Contact uses personal email (Gmail/Yahoo) — not official'})
    if len(risk_flags) > 2:
        flags.append({'type': 'danger', 'text': f'Suspicious keywords detected: {", ".join(risk_flags[:4])}'})
    elif len(risk_flags) > 0:
        flags.append({'type': 'warn', 'text': f'Potentially suspicious phrase detected: {risk_flags[0]}'})
    if has_official_url:
        flags.append({'type': 'safe', 'text': 'URL appears to be an official government/education domain'})
    if has_official_email:
        flags.append({'type': 'safe', 'text': 'Contact email from an official domain (.gov/.edu/.ac.in)'})
    if len(safe_flags) >= 2:
        flags.append({'type': 'safe', 'text': f'Official indicators found: {", ".join(safe_flags[:3])}'})
    if not flags:
        flags.append({'type': 'warn', 'text': 'Insufficient information — verify with official sources'})

    return {
        'predictions':      predictions,
        'ensemble_label':   ensemble_label,
        'ensemble_conf':    ensemble_conf,
        'votes_fake':       fake_votes,
        'votes_real':       real_votes,
        'votes_total':      total,
        'flags':            flags,
        'suspicious_kws':   risk_flags[:6],
        'safe_kws':         safe_flags[:6],
    }


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    meta_stats = META.get('results', {})
    return render_template('index.html',
                           models_loaded=len(MODELS),
                           total_models=len(MODEL_KEYS),
                           meta=META,
                           meta_stats=meta_stats)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data received'}), 400

    required = ['scholarship_name', 'provider', 'description']
    for field in required:
        if not data.get(field, '').strip():
            return jsonify({'error': f'Missing required field: {field}'}), 400

    feature_text = build_feature(data)
    result = predict_all(feature_text)

    return jsonify({
        'success': True,
        'input':   {k: data.get(k, '') for k in ['scholarship_name','provider','description',
                                                   'url','contact_email','application_fee','amount']},
        'result':  result,
    })


@app.route('/models')
def models_info():
    """JSON endpoint with model training metadata."""
    return jsonify(META)


@app.route('/dataset')
def dataset_info():
    """Return dataset summary."""
    import pandas as pd
    try:
        df = pd.read_csv(os.path.join(BASE, 'dataset', 'scholarship_dataset.csv'))
        return jsonify({
            'total': len(df),
            'real': int((df['label'] == 'REAL').sum()),
            'fake': int((df['label'] == 'FAKE').sum()),
            'columns': list(df.columns),
            'sample_real': df[df['label']=='REAL'][['scholarship_name','provider']].head(5).to_dict('records'),
            'sample_fake': df[df['label']=='FAKE'][['scholarship_name','provider']].head(5).to_dict('records'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n🛡️  ScholarShield — Starting Flask Server")
    print(f"   Models loaded: {len(MODELS)}/{len(MODEL_KEYS)}")
    print("   URL: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
