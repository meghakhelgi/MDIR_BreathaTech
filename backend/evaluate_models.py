"""
evaluate_models.py — BreathaTech model evaluation figures

Generates ROC curves, PR curves, confusion matrices, feature importance,
and a classification report for the agent and severity classifiers.
Handles the cascade architecture: severity model receives agent probability
outputs as additional features, exactly as during training.

Usage:
    cd backend
    python evaluate_models.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve

from train_models import (
    engineer_features,
    FEATURES, SEV_FEATURES, SENSOR_FEATURES, AGENT_CLASSES,
)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'training_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUT_DIR   = os.path.join(BASE_DIR, 'evaluation')
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette matching the UI ─────────────────────────────────────────────
AGENT_COLORS = {
    'CO':       '#1d6fb8',
    'NONE':     '#111111',
    'OP':       '#dc2626',
    'PHOSGENE': '#d97706',
}
SEV_COLORS = {
    '0': '#94a3b8',
    '1': '#059669',
    '2': '#d97706',
    '3': '#dc2626',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


# ── helpers ────────────────────────────────────────────────────────────────────

def load_pkl(name):
    with open(os.path.join(MODEL_DIR, name), 'rb') as f:
        return pickle.load(f)


def plot_roc(ax, y_true_bin, y_prob, classes, colors, title):
    """One-vs-rest ROC curve per class + macro average."""
    macro_tpr = np.linspace(0, 1, 300)
    tprs = []
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        cls_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors.get(str(cls), '#94a3b8'),
                lw=1.8, label=f'{cls}  (AUC {cls_auc:.3f})')
        tprs.append(np.interp(macro_tpr, fpr, tpr))

    macro_tpr_mean = np.mean(tprs, axis=0)
    macro_auc = auc(macro_tpr, macro_tpr_mean)
    ax.plot(macro_tpr, macro_tpr_mean, 'k--', lw=1.4,
            label=f'Macro avg  (AUC {macro_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#cbd5e1', lw=1, ls=':')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title=title)
    ax.legend(fontsize=8, loc='lower right')


def plot_pr(ax, y_true_bin, y_prob, classes, colors, title):
    """One-vs-rest PR curve per class + macro average AP."""
    for i, cls in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        ax.plot(rec, prec, color=colors.get(str(cls), '#94a3b8'),
                lw=1.8, label=f'{cls}  (AP {ap:.3f})')
    ax.set(xlabel='Recall', ylabel='Precision', title=title, ylim=[0, 1.05])
    ax.legend(fontsize=8, loc='lower left')


def plot_confusion(ax, y_true, y_pred, classes, title):
    """Normalised confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'shrink': 0.8}, linewidths=0.5, linecolor='#e2e8f0',
                vmin=0, vmax=1)
    ax.set(xlabel='Predicted', ylabel='Actual', title=title)
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)


def plot_importance(ax, model, title, top_n=20):
    """Top-N features by XGBoost gain."""
    scores = model.get_booster().get_score(importance_type='gain')
    s = pd.Series(scores).nlargest(top_n).sort_values()
    colors = ['#1d6fb8' if v > s.median() else '#93c5fd' for v in s]
    s.plot.barh(ax=ax, color=colors, edgecolor='none')
    ax.set(xlabel='Gain', title=title)
    ax.tick_params(labelsize=8)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    X  = df[FEATURES]

    agent_model    = load_pkl('agent_model.pkl')
    severity_model = load_pkl('severity_model.pkl')
    agent_le       = load_pkl('agent_le.pkl')
    severity_le    = load_pkl('severity_le.pkl')

    # ── Agent test split (reproduces train_models.py exactly) ─────────────────
    y_agent = agent_le.transform(df['agent'])
    _, X_te_a, _, ya_te = train_test_split(
        X, y_agent, test_size=0.2, random_state=42, stratify=y_agent
    )
    ya_pred  = agent_model.predict(X_te_a)
    ya_prob  = agent_model.predict_proba(X_te_a)
    a_classes = agent_le.classes_          # ['CO', 'NONE', 'OP', 'PHOSGENE']
    ya_bin   = label_binarize(ya_te, classes=np.arange(len(a_classes)))

    # ── Sensor-only ablation ───────────────────────────────────────────────────
    from xgboost import XGBClassifier
    X_tr_a, X_te_a_full, ya_tr, _ = train_test_split(
        X, y_agent, test_size=0.2, random_state=42, stratify=y_agent
    )
    sensor_model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1,
    )
    sensor_model.fit(X_tr_a[SENSOR_FEATURES], ya_tr)
    sensor_prob = sensor_model.predict_proba(X_te_a[SENSOR_FEATURES])
    sensor_auc  = roc_auc_score(ya_te, sensor_prob, multi_class='ovr', average='macro')

    agent_auc = roc_auc_score(ya_te, ya_prob, multi_class='ovr', average='macro')

    # ── Cascade: build SEV_FEATURES for severity test split ───────────────────
    # Compute agent_prob_* on the full dataset (same as train_models.py)
    ya_prob_all = agent_model.predict_proba(X)
    df_sev = df.copy()
    for i, cls in enumerate(AGENT_CLASSES):
        df_sev[f'agent_prob_{cls}'] = ya_prob_all[:, i]

    X_sev_full = df_sev[SEV_FEATURES]
    y_sev = severity_le.transform(df['severity'])

    _, X_te_s, _, ys_te = train_test_split(
        X_sev_full, y_sev, test_size=0.2, random_state=42, stratify=y_sev
    )
    ys_pred   = severity_model.predict(X_te_s)
    ys_prob   = severity_model.predict_proba(X_te_s)
    s_classes = severity_le.classes_
    ys_bin    = label_binarize(ys_te, classes=np.arange(len(s_classes)))

    # ── Classification reports ─────────────────────────────────────────────────
    print("\n-- Agent classifier ------------------------------------------")
    print(classification_report(
        ya_te, ya_pred,
        target_names=a_classes,
        digits=3
    ))
    print(f"Macro OvR AUC (full features):  {agent_auc:.4f}")
    print(f"Macro OvR AUC (sensors only):   {sensor_auc:.4f}")
    print(f"Clinical data uplift:           +{agent_auc - sensor_auc:.4f}")

    print("\n-- Severity classifier (cascade) ------------------------------")
    print(classification_report(
        ys_te, ys_pred,
        target_names=[str(c) for c in s_classes],
        digits=3
    ))
    sev_macro_auc = roc_auc_score(ys_te, ys_prob, multi_class='ovr', average='macro')
    print(f"Macro OvR AUC: {sev_macro_auc:.4f}")

    # ── Figure 1: ROC curves ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ROC Curves — One-vs-Rest', fontweight='bold', fontsize=12)

    plot_roc(axes[0], ya_bin, ya_prob, a_classes, AGENT_COLORS,
             'Agent Classifier')
    plot_roc(axes[1], ys_bin, ys_prob,
             [str(c) for c in s_classes], SEV_COLORS,
             'Severity Classifier (cascade)')

    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'fig1_roc_curves.png')
    fig.savefig(p, bbox_inches='tight')
    print(f"\nSaved -> {p}")
    plt.close(fig)

    # ── Figure 2: Precision-Recall curves ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Precision-Recall Curves — One-vs-Rest', fontweight='bold', fontsize=12)

    plot_pr(axes[0], ya_bin, ya_prob, a_classes, AGENT_COLORS,
            'Agent Classifier')
    plot_pr(axes[1], ys_bin, ys_prob,
            [str(c) for c in s_classes], SEV_COLORS,
            'Severity Classifier (cascade)')

    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'fig2_pr_curves.png')
    fig.savefig(p, bbox_inches='tight')
    print(f"Saved -> {p}")
    plt.close(fig)

    # ── Figure 3: Confusion matrices ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Confusion Matrices (normalised by true class)', fontweight='bold', fontsize=12)

    ya_pred_labels = agent_le.inverse_transform(ya_pred)
    ya_true_labels = agent_le.inverse_transform(ya_te)
    plot_confusion(axes[0], ya_true_labels, ya_pred_labels, list(a_classes),
                   'Agent Classifier')

    ys_pred_labels = severity_le.inverse_transform(ys_pred)
    ys_true_labels = severity_le.inverse_transform(ys_te)
    plot_confusion(axes[1], ys_true_labels, ys_pred_labels,
                   list(s_classes), 'Severity Classifier (cascade)')

    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'fig3_confusion_matrices.png')
    fig.savefig(p, bbox_inches='tight')
    print(f"Saved -> {p}")
    plt.close(fig)

    # ── Figure 4: Feature importance ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Feature Importance — XGBoost Gain (top 20)', fontweight='bold', fontsize=12)

    plot_importance(axes[0], agent_model,    'Agent Classifier')
    plot_importance(axes[1], severity_model, 'Severity Classifier (cascade)')

    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'fig4_feature_importance.png')
    fig.savefig(p, bbox_inches='tight')
    print(f"Saved -> {p}")
    plt.close(fig)

    # ── Figure 5: Sensor-only ablation ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle('Sensor-Only Ablation — Agent Classifier', fontweight='bold', fontsize=12)

    sensor_bin = label_binarize(ya_te, classes=np.arange(len(a_classes)))
    macro_tpr = np.linspace(0, 1, 300)

    for label, probs, ls in [('Full features', ya_prob, '-'), ('Sensors only', sensor_prob, '--')]:
        tprs = []
        for i in range(len(a_classes)):
            fpr, tpr, _ = roc_curve(sensor_bin[:, i], probs[:, i])
            tprs.append(np.interp(macro_tpr, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(macro_tpr, mean_tpr)
        ax.plot(macro_tpr, mean_tpr, ls=ls, lw=2,
                label=f'{label}  (macro AUC {mean_auc:.3f})')

    ax.plot([0, 1], [0, 1], color='#cbd5e1', lw=1, ls=':')
    ax.set(xlabel='False positive rate', ylabel='True positive rate',
           title='Macro-avg ROC: full vs. sensor-only')
    ax.legend(fontsize=9)

    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'fig5_sensor_ablation.png')
    fig.savefig(p, bbox_inches='tight')
    print(f"Saved -> {p}")
    plt.close(fig)

    # ── Figure 6: t-SNE — agent class separation in feature space ─────────────
    print("\nRunning t-SNE (this takes ~30 s) …")

    # Track which rows end up in the test split so we can overlay misclassified pts
    idx_all = np.arange(len(df))
    _, idx_te = train_test_split(idx_all, test_size=0.2, random_state=42, stratify=y_agent)
    ya_pred_all = agent_model.predict(X)
    miscls_mask = np.zeros(len(df), dtype=bool)
    miscls_mask[idx_te] = ya_pred_all[idx_te] != y_agent[idx_te]

    X_2d = TSNE(n_components=2, perplexity=40, random_state=42, n_jobs=-1).fit_transform(X.values)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls in a_classes:
        mask = df['agent'].values == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=AGENT_COLORS[cls], label=cls, alpha=0.45, s=18, lw=0)
    ax.scatter(X_2d[miscls_mask, 0], X_2d[miscls_mask, 1],
               c='black', marker='x', s=55, lw=1.2, zorder=5,
               label='Misclassified (test)')
    ax.set(title='t-SNE — agent feature space  (× = test-set error)',
           xlabel='t-SNE 1', ylabel='t-SNE 2')
    ax.legend(fontsize=9, markerscale=1.4)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'fig6_tsne.png')
    fig.savefig(p, bbox_inches='tight')
    print(f"Saved -> {p}")
    plt.close(fig)

    # ── Figure 7: Calibration curves ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Calibration Curves — One-vs-Rest\n'
                 '(perfect calibration = dashed diagonal)',
                 fontweight='bold', fontsize=12)

    # Agent calibration
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect')
    for i, cls in enumerate(a_classes):
        frac_pos, mean_pred = calibration_curve(
            (ya_te == i).astype(int), ya_prob[:, i], n_bins=10, strategy='uniform'
        )
        ax.plot(mean_pred, frac_pos, marker='o', ms=5,
                color=AGENT_COLORS[cls], lw=1.8, label=cls)
    ax.set(xlabel='Mean predicted probability', ylabel='Fraction positive',
           title='Agent Classifier', xlim=[0, 1], ylim=[0, 1])
    ax.legend(fontsize=8)

    # Severity calibration
    ax = axes[1]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect')
    for i, cls in enumerate(s_classes):
        frac_pos, mean_pred = calibration_curve(
            (ys_te == i).astype(int), ys_prob[:, i], n_bins=10, strategy='uniform'
        )
        ax.plot(mean_pred, frac_pos, marker='o', ms=5,
                color=SEV_COLORS[str(cls)], lw=1.8, label=f'Severity {cls}')
    ax.set(xlabel='Mean predicted probability', ylabel='Fraction positive',
           title='Severity Classifier (cascade)', xlim=[0, 1], ylim=[0, 1])
    ax.legend(fontsize=8)

    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'fig7_calibration.png')
    fig.savefig(p, bbox_inches='tight')
    print(f"Saved -> {p}")
    plt.close(fig)

    # ── Figure 8: Sensor reading distributions by agent ────────────────────────
    sensor_cols  = ['eco_ppm', 'eno_ppb', 'eco2_pct', 'op_score']
    sensor_labels = ['eCO (ppm)', 'eNO (ppb)', 'eCO₂ (%)', 'OP Score']
    agent_order  = ['CO', 'NONE', 'OP', 'PHOSGENE']

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle('Sensor Reading Distributions by Agent Class',
                 fontweight='bold', fontsize=12)

    plot_df = df[['agent'] + sensor_cols].copy()
    palette  = {cls: AGENT_COLORS[cls] for cls in agent_order}

    for ax, col, label in zip(axes, sensor_cols, sensor_labels):
        sns.violinplot(
            data=plot_df, x='agent', y=col, hue='agent', order=agent_order,
            palette=palette, ax=ax, inner='quartile', linewidth=0.8,
            cut=0, legend=False,
        )
        ax.set(xlabel='', ylabel=label, title=label)
        ax.tick_params(axis='x', rotation=20, labelsize=8)

    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'fig8_sensor_distributions.png')
    fig.savefig(p, bbox_inches='tight')
    print(f"Saved -> {p}")
    plt.close(fig)

    # ── Figure 9: Sensor pair plot — where the model actually discriminates ──────
    sensor_raw = ['eco_ppm', 'eno_ppb', 'eco2_pct', 'op_score']
    sensor_axis_labels = {
        'eco_ppm':   'eCO (ppm)',
        'eno_ppb':   'eNO (ppb)',
        'eco2_pct':  'eCO₂ (%)',
        'op_score':  'OP Score',
    }
    agent_order = ['CO', 'NONE', 'OP', 'PHOSGENE']
    pairs = [(sensor_raw[i], sensor_raw[j])
             for i in range(len(sensor_raw))
             for j in range(i + 1, len(sensor_raw))]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Sensor Pair Plot — Agent Discrimination in Raw Sensor Space',
                 fontweight='bold', fontsize=12)

    for ax, (xcol, ycol) in zip(axes.flat, pairs):
        for cls in agent_order:
            mask = df['agent'].values == cls
            ax.scatter(df.loc[mask, xcol], df.loc[mask, ycol],
                       c=AGENT_COLORS[cls], label=cls,
                       alpha=0.35, s=12, lw=0)
        ax.set(xlabel=sensor_axis_labels[xcol], ylabel=sensor_axis_labels[ycol])
        ax.tick_params(labelsize=8)

    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=AGENT_COLORS[c], markersize=8, label=c)
               for c in agent_order]
    fig.legend(handles=handles, loc='lower right', fontsize=10,
               bbox_to_anchor=(0.98, 0.02))
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'fig9_sensor_pairs.png')
    fig.savefig(p, bbox_inches='tight')
    print(f"Saved -> {p}")
    plt.close(fig)

    # ── Figure 10: Predicted probability space — how the classifier sees the data ─
    # Each patient is a point in 4D probability space (sums to 1).
    # We plot all 6 pairwise combinations. Well-classified patients cluster near
    # (1,0) in their true-class axis; uncertain/confused cases scatter toward centre.
    ya_prob_full = agent_model.predict_proba(X)
    prob_df = pd.DataFrame(ya_prob_full, columns=[f'P({c})' for c in a_classes])
    prob_df['true_agent'] = agent_le.inverse_transform(y_agent)

    pairs_prob = [(f'P({a_classes[i]})', f'P({a_classes[j]})')
                  for i in range(len(a_classes))
                  for j in range(i + 1, len(a_classes))]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Predicted Probability Space — How the Classifier Sees Each Patient\n'
                 '(correctly classified patients cluster near corners; '
                 'confused cases scatter toward centre)',
                 fontweight='bold', fontsize=11)

    for ax, (xcol, ycol) in zip(axes.flat, pairs_prob):
        for cls in agent_order:
            mask = prob_df['true_agent'] == cls
            ax.scatter(prob_df.loc[mask, xcol], prob_df.loc[mask, ycol],
                       c=AGENT_COLORS[cls], label=cls,
                       alpha=0.35, s=12, lw=0)
        ax.set(xlabel=xcol, ylabel=ycol, xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.tick_params(labelsize=8)

    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=AGENT_COLORS[c], markersize=8, label=c)
               for c in agent_order]
    fig.legend(handles=handles, loc='lower right', fontsize=10,
               bbox_to_anchor=(0.98, 0.02))
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'fig10_probability_space.png')
    fig.savefig(p, bbox_inches='tight')
    print(f"Saved -> {p}")
    plt.close(fig)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
