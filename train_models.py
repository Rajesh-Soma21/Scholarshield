"""
train_models.py
================
Trains and saves all 4 ML models for the Fake Scholarship Detection System.
Run this once before starting the Flask app.

Models trained:
  1. Naive Bayes (MultinomialNB)
  2. Logistic Regression
  3. Decision Tree
  4. Support Vector Machine (LinearSVC)

Dataset: dataset/scholarship_dataset.csv (200+ labeled records)
Feature Engineering: TF-IDF (unigrams + bigrams, top 5000 features)
"""

import os, json, joblib, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)

np.random.seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. LOAD DATASET ────────────────────────────────────────────────────────────
print("=" * 65)
print("  ScholarShield — Model Training Pipeline")
print("  Fake Scholarship Detection System")
print("  Student: Divya | USN: U18IW23S0016")
print("=" * 65)

df = pd.read_csv(os.path.join(BASE, 'dataset', 'scholarship_dataset.csv'))
print(f"\n📂 Dataset loaded: {len(df)} records")
print(f"   REAL : {(df.label=='REAL').sum()}")
print(f"   FAKE : {(df.label=='FAKE').sum()}")

# ── 2. FEATURE ENGINEERING ─────────────────────────────────────────────────────
# Combine all text fields into one rich feature string
def build_feature(row):
    parts = [
        str(row.get('scholarship_name', '')),
        str(row.get('provider', '')),
        str(row.get('description', '')),
        str(row.get('url', '')),
        str(row.get('contact_email', '')),
        str(row.get('application_fee', '')),
        str(row.get('amount', ''))
    ]
    return ' '.join(p for p in parts if p and p.lower() != 'nan')

df['combined_text'] = df.apply(build_feature, axis=1)
df['label_bin'] = (df['label'] == 'FAKE').astype(int)

X = df['combined_text']
y = df['label_bin']

# ── 3. TRAIN / TEST SPLIT ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n✂️  Split — Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── 4. DEFINE PIPELINES ────────────────────────────────────────────────────────
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True,
    min_df=1,
    strip_accents='unicode',
    analyzer='word'
)

pipelines = {
    'naive_bayes': Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf',   MultinomialNB(alpha=0.3))
    ]),
    'logistic_regression': Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf',   LogisticRegression(max_iter=1000, C=1.5, solver='lbfgs', random_state=42))
    ]),
    'decision_tree': Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf',   DecisionTreeClassifier(max_depth=15, min_samples_split=3,
                                         min_samples_leaf=1, random_state=42))
    ]),
    'svm': Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf',   LinearSVC(C=1.2, max_iter=3000, random_state=42))
    ]),
}

DISPLAY_NAMES = {
    'naive_bayes': 'Naive Bayes',
    'logistic_regression': 'Logistic Regression',
    'decision_tree': 'Decision Tree',
    'svm': 'SVM (LinearSVC)',
}

# ── 5. TRAIN, EVALUATE, SAVE ───────────────────────────────────────────────────
results = {}
cms = {}

print("\n─── Training Models ─────────────────────────────────────────\n")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for key, pipe in pipelines.items():
    dname = DISPLAY_NAMES[key]
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cv_s = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')

    results[dname] = {
        'accuracy':  round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall':    round(rec * 100, 2),
        'f1_score':  round(f1 * 100, 2),
        'cv_mean':   round(cv_s.mean() * 100, 2),
        'cv_std':    round(cv_s.std() * 100, 2),
    }
    cms[dname] = confusion_matrix(y_test, y_pred).tolist()

    # Save pipeline
    save_path = os.path.join(MODELS_DIR, f'{key}.pkl')
    joblib.dump(pipe, save_path)

    print(f"  ✅ {dname}")
    print(f"     Accuracy   : {acc*100:.2f}%  |  CV: {cv_s.mean()*100:.2f}% ± {cv_s.std()*100:.2f}%")
    print(f"     Precision  : {prec*100:.2f}%  |  Recall: {rec*100:.2f}%  |  F1: {f1*100:.2f}%")
    print(f"     Saved to   : {save_path}")
    print()

# ── 6. SAVE METADATA ──────────────────────────────────────────────────────────
meta = {
    'models': list(pipelines.keys()),
    'display_names': DISPLAY_NAMES,
    'results': results,
    'confusion_matrices': cms,
    'dataset_size': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'feature_method': 'TF-IDF (unigrams + bigrams, max 5000 features)',
    'label_mapping': {'0': 'REAL', '1': 'FAKE'},
}
with open(os.path.join(MODELS_DIR, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
print(f"  📄 Metadata saved: models/meta.json")

# ── 7. GENERATE CHARTS ────────────────────────────────────────────────────────
BG = '#050a14'; SURF = '#0b1526'; SURF2 = '#111e34'
CYAN = '#00e5ff'; GREEN = '#00ff9d'; AMBER = '#ffb300'; RED = '#ff4444'
MUTED = '#7a95b4'; TEXT = '#e8f4fd'
COLORS = [CYAN, GREEN, AMBER, RED]

plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle('ScholarShield — ML Model Evaluation Results\nStudent: Divya | USN: U18IW23S0016',
             fontsize=14, color=TEXT, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35)

model_names = list(results.keys())
short = ['Naive\nBayes', 'Logistic\nRegression', 'Decision\nTree', 'SVM']
accs  = [results[m]['accuracy']  for m in model_names]
precs = [results[m]['precision'] for m in model_names]
recs  = [results[m]['recall']    for m in model_names]
f1s   = [results[m]['f1_score']  for m in model_names]
cvs   = [results[m]['cv_mean']   for m in model_names]
cvstd = [results[m]['cv_std']    for m in model_names]

def style_ax(ax):
    ax.set_facecolor(SURF)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values(): sp.set_color('#1d3557')
    ax.grid(axis='y', alpha=0.15, color=MUTED, linestyle='--')

# Plot 1: grouped bars
ax1 = fig.add_subplot(gs[0, :2])
x = np.arange(4); w = 0.18
for i, (vals, lbl, c) in enumerate(zip([accs, precs, recs, f1s],
                                        ['Accuracy','Precision','Recall','F1-Score'],
                                        [CYAN, GREEN, AMBER, '#b47aff'])):
    bars = ax1.bar(x + (i-1.5)*w, vals, w, label=lbl, color=c, alpha=0.9)
    for bar in bars:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{bar.get_height():.1f}', ha='center', va='bottom',
                 fontsize=6.5, color=TEXT, fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(short, color=TEXT, fontsize=9)
ax1.set_ylim(0, 118); ax1.set_ylabel('Score (%)', color=MUTED, fontsize=9)
ax1.set_title('Performance Metrics Comparison', color=TEXT, fontsize=10, pad=8)
ax1.legend(fontsize=7, labelcolor=TEXT, framealpha=0.2)
style_ax(ax1)

# Plot 2: CV accuracy
ax2 = fig.add_subplot(gs[0, 2:])
bars = ax2.bar(short, cvs, color=COLORS, alpha=0.88, yerr=cvstd,
               capsize=5, error_kw={'color': TEXT, 'elinewidth': 1.5, 'capthick': 1.5})
for bar, v in zip(bars, cvs):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
             f'{v:.1f}%', ha='center', va='bottom', fontsize=9, color=TEXT, fontweight='bold')
ax2.set_ylim(0, 118); ax2.set_ylabel('CV Accuracy (%)', color=MUTED, fontsize=9)
ax2.set_title('5-Fold Cross Validation Accuracy', color=TEXT, fontsize=10, pad=8)
style_ax(ax2)

# Plots 3-6: confusion matrices
for idx, mname in enumerate(model_names):
    row = 1 + idx // 2; col = (idx % 2) * 2
    ax = fig.add_subplot(gs[row, col:col+2])
    cm = np.array(cms[mname])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['REAL','FAKE'], yticklabels=['REAL','FAKE'],
                annot_kws={'size': 13, 'weight': 'bold'},
                linewidths=0.5, linecolor='#1d3557',
                cbar_kws={'shrink': 0.75})
    ax.set_title(f'{mname}', color=TEXT, fontsize=9, pad=6)
    ax.set_xlabel('Predicted', color=MUTED, fontsize=8)
    ax.set_ylabel('Actual', color=MUTED, fontsize=8)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.text(0.98, 0.02, f'Acc: {results[mname]["accuracy"]:.1f}%',
            transform=ax.transAxes, ha='right', va='bottom',
            color=CYAN, fontsize=9, fontweight='bold')

chart_path = os.path.join(BASE, 'static', 'img', 'ml_results.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close()
print(f"  📊 Chart saved: static/img/ml_results.png")

print("\n" + "=" * 65)
print("  ✅ ALL MODELS TRAINED & SAVED SUCCESSFULLY")
print(f"  📁 Model files in: {MODELS_DIR}/")
print("=" * 65)
