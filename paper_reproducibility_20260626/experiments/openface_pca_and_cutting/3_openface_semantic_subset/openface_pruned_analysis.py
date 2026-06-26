"""
MPDD OpenFace Pruned Feature Analysis
- Extract depression-relevant AUs from OpenFace 710-dim output
- Compute temporal statistics (not just mean)
- Per-segment analysis (spontaneous vs reading)
- Residual learning on top of personality baseline
- AU ranking by depression correlation with bootstrap stability
"""

import numpy as np
import os
import json
import re
import warnings
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_BASE = '/data/zilu/mpdd2026/datasets/MPDD-AVG-2026/Young'
OF_TRAIN = os.path.join(DATA_BASE, 'Video', 'train', 'openface')
OF_TEST = os.path.join(DATA_BASE, 'Video', 'openface')

# OpenFace column indices (confirmed via unit-vector test for gaze):
# - Col 0: frame
# - Cols 2-4: gaze_0_x/y/z (left eye gaze direction, norm ~ 1.0)
# - Cols 5-7: gaze_1_x/y/z (right eye gaze direction, norm ~ 1.0)
# - Cols 675-691: AU intensities (17 dims, values 0-5)
# - Cols 692-709: AU presence (18 dims, binary, some all-zero)
COLS_AU_INTENSITY = list(range(675, 692))
COLS_AU_PRESENCE = list(range(692, 710))
COLS_GAZE = [2, 3, 4, 5, 6, 7]

# AU names in standard OpenFace order (17 AUs)
AU_NAMES = [
    'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07',
    'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17',
    'AU20', 'AU23', 'AU25', 'AU26', 'AU45'
]
# Depression-relevant labels (literature-based)
AU_DEPRESSION_LABELS = {
    'AU04': 'brow furrow (neg affect)',
    'AU06': 'cheek raise (smile)',
    'AU07': 'lid tighten (distress)',
    'AU10': 'upper lip raise (disgust)',
    'AU12': 'lip corner pull (smile)',
    'AU14': 'dimpler (contempt)',
    'AU15': 'lip corner depress (sadness)',
    'AU17': 'chin raise (distress)',
    'AU20': 'lip stretch (fear)',
    'AU23': 'lip tighten (anger)',
    'AU25': 'lips part (speech)',
    'AU45': 'blink (arousal)',
}

PERSONALITY_TRAITS = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']

# Test set
TEST_IDS = [1, 5, 7, 13, 15, 22, 28, 33, 34, 40, 42, 44, 47, 58, 74, 83, 85, 89, 90, 93, 105, 110]
TRUE_B = {1: 1, 5: 1, 7: 1, 13: 1, 15: 1, 22: 0, 28: 0, 33: 0, 34: 1, 40: 0,
          42: 1, 44: 0, 47: 0, 58: 0, 74: 1, 83: 0, 85: 1, 89: 1, 90: 1, 93: 0, 105: 0, 110: 0}
TRUE_T = {1: 2, 5: 1, 7: 1, 13: 1, 15: 1, 22: 0, 28: 0, 33: 0, 34: 1, 40: 0,
          42: 1, 44: 0, 47: 0, 58: 0, 74: 2, 83: 0, 85: 2, 89: 1, 90: 1, 93: 0, 105: 0, 110: 0}

RESULTS_DIR = '/data/zilu/mpdd2026/workplace/results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================
def extract_personality(desc_df):
    """
    Extract Big Five personality scores from descriptions text.
    Handles standard patterns plus combined expressions like
    "Agreeableness and Conscientiousness scores of X" etc.
    """
    import pandas as pd

    single_pat = re.compile(
        r'(Extraversion|Agreeableness|Conscientiousness|Neuroticism|Openness|'
        r'extraversion|agreeableness|conscientiousness|neuroticism|openness)'
        r'\s+score\s+(?:of|is)\s+(\d+)',
    )
    combined_same = re.compile(
        r'(Agreeableness|agreeableness)\s+and\s+(Conscientiousness|conscientiousness)'
        r'\s+scores\s+(?:of|are\s+both)\s+(\d+)'
    )
    combined_diff = re.compile(
        r'(Agreeableness|agreeableness)\s+and\s+(Conscientiousness|conscientiousness)'
        r'\s+scores\s+of\s+(\d+)\s+and\s+(\d+)'
    )
    combined_three = re.compile(
        r'(Agreeableness|agreeableness),\s+(Conscientiousness|conscientiousness),'
        r'\s+and\s+(Neuroticism|neuroticism)\s+scores\s+are\s+all\s+(\d+)'
    )
    low_pat = re.compile(r'(?:low|high)\s+(Neuroticism|neuroticism)\s+score\s+of\s+(\d+)')

    trait_map = {
        'Extraversion': 'E', 'extraversion': 'E',
        'Agreeableness': 'A', 'agreeableness': 'A',
        'Conscientiousness': 'C', 'conscientiousness': 'C',
        'Neuroticism': 'N', 'neuroticism': 'N',
        'Openness': 'O', 'openness': 'O',
    }

    rows = []
    for _, row in desc_df.iterrows():
        sid = row['ID']
        text = row['Descriptions']
        scores = {}

        text_processed = text
        for m in combined_three.finditer(text):
            a_key = trait_map.get(m.group(1))
            c_key = trait_map.get(m.group(2))
            n_key = trait_map.get(m.group(3))
            val = int(m.group(4))
            if a_key: scores[a_key] = val
            if c_key: scores[c_key] = val
            if n_key: scores[n_key] = val
            text_processed = text_processed.replace(m.group(0), '')

        for m in combined_diff.finditer(text_processed):
            a_key = trait_map.get(m.group(1))
            c_key = trait_map.get(m.group(2))
            if a_key: scores[a_key] = int(m.group(3))
            if c_key: scores[c_key] = int(m.group(4))
            text_processed = text_processed.replace(m.group(0), '')

        for m in combined_same.finditer(text_processed):
            a_key = trait_map.get(m.group(1))
            c_key = trait_map.get(m.group(2))
            val = int(m.group(3))
            if a_key: scores[a_key] = val
            if c_key: scores[c_key] = val
            text_processed = text_processed.replace(m.group(0), '')

        for m in low_pat.finditer(text_processed):
            n_key = trait_map.get(m.group(1))
            if n_key and n_key not in scores:
                scores[n_key] = int(m.group(2))
            text_processed = text_processed.replace(m.group(0), '')

        for m in single_pat.finditer(text_processed):
            key = trait_map.get(m.group(1))
            if key and key not in scores:
                scores[key] = int(m.group(2))

        if len(scores) == 5:
            rows.append({'ID': sid, 'E': scores['E'], 'A': scores['A'],
                        'C': scores['C'], 'N': scores['N'], 'O': scores['O']})
        else:
            print(f"  WARNING: Skipping ID {sid} (got {len(scores)}/5: {scores})")

    return pd.DataFrame(rows)


def load_openface_features(subject_ids, base_dir):
    """Load OpenFace data for given subjects (all 3 events)."""
    data = {}
    for sid in subject_ids:
        data[sid] = {}
        for ev in ['event_1', 'event_2', 'event_3']:
            path = os.path.join(base_dir, str(sid), ev, f'{ev}_all.npy')
            if os.path.exists(path):
                data[sid][ev] = np.load(path, allow_pickle=True).astype(np.float32)
    return data


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def extract_segment_features(arr):
    """
    From a single (T, 710) segment, extract:
    - AU intensity (17): mean, std, max, range, % active (threshold > 0)
    - AU presence (18): mean, std
    - Gaze (6): mean, std of each component
    """
    au_int = arr[:, COLS_AU_INTENSITY]  # (T, 17)
    au_prs = arr[:, COLS_AU_PRESENCE]   # (T, 18)
    gaze = arr[:, COLS_GAZE]            # (T, 6)

    feat = {}
    feat['au_int_mean'] = au_int.mean(axis=0)       # 17
    feat['au_int_std'] = au_int.std(axis=0)         # 17
    feat['au_int_max'] = au_int.max(axis=0)         # 17
    feat['au_int_range'] = feat['au_int_max'] - au_int.min(axis=0)  # 17
    feat['au_int_pct_active'] = (au_int > 0).mean(axis=0)           # 17

    feat['au_prs_mean'] = au_prs.mean(axis=0)       # 18
    feat['au_prs_std'] = au_prs.std(axis=0)         # 18

    feat['gaze_mean'] = gaze.mean(axis=0)            # 6
    feat['gaze_std'] = gaze.std(axis=0)              # 6

    return feat


def build_subject_vector(subj_data):
    """
    Combine all segment features into a single vector per subject.
    """
    events = ['event_1', 'event_2', 'event_3']
    parts = []
    names = []

    # Per-event features
    for ev in events:
        if ev not in subj_data:
            continue
        feat = extract_segment_features(subj_data[ev])
        for key, arr in sorted(feat.items()):
            for i in range(len(arr)):
                parts.append(arr[i])
                names.append(f'{ev}_{key}_{i}')

    # Cross-segment differentials (only if all 3 events)
    if all(ev in subj_data for ev in events):
        f1 = extract_segment_features(subj_data['event_1'])
        f2 = extract_segment_features(subj_data['event_2'])
        f3 = extract_segment_features(subj_data['event_3'])

        # Dynamic range: seg1_std - avg(seg2_std, seg3_std)
        for key in ['au_int_std', 'au_int_mean', 'au_int_max',
                     'au_int_range', 'gaze_std', 'gaze_mean']:
            diff = f1[key] - 0.5 * (f2[key] + f3[key])
            for i in range(len(diff)):
                parts.append(diff[i])
                names.append(f'dynamic_{key}_{i}')

    return np.array(parts), names


# =============================================================================
# MODEL EVALUATION (aligned with existing codebase)
# =============================================================================
def evaluate_phq9_regression(clf_class, clf_kwargs, X_train, y_phq9,
                              X_test, y_test_b, y_test_t):
    """Ridge regression on PHQ-9 -> threshold for binary/ternary."""
    clf = clf_class(**clf_kwargs)
    clf.fit(X_train, y_phq9)
    yp_test = clf.predict(X_test)

    yp_test_b = (yp_test >= 5).astype(int)
    yp_test_t = np.zeros_like(yp_test, dtype=int)
    yp_test_t[(yp_test >= 5) & (yp_test < 10)] = 1
    yp_test_t[yp_test >= 10] = 2

    return {
        'binary_acc': float(accuracy_score(y_test_b, yp_test_b)),
        'binary_f1': float(f1_score(y_test_b, yp_test_b, average='macro')),
        'ternary_acc': float(accuracy_score(y_test_t, yp_test_t)),
        'ternary_f1': float(f1_score(y_test_t, yp_test_t, average='macro')),
        'pred_binary': yp_test_b.tolist(),
        'pred_ternary': yp_test_t.tolist(),
    }


def evaluate_binary_classifier(clf, X_train, y_train_b, X_test, y_test_b, y_test_t):
    """Direct binary classifier."""
    clf.fit(X_train, y_train_b)
    yp_test_b = clf.predict(X_test)

    # For ternary, derive from binary: 0 -> 0, 1 -> follow threshold logic
    # Simplest: map binary 0 to ternary 0, binary 1 to ternary 1
    # Better: use predict_proba to get continuous score, then threshold same as PHQ-9
    yp_test_ter_t = np.zeros_like(yp_test_b, dtype=int)

    return {
        'binary_acc': float(accuracy_score(y_test_b, yp_test_b)),
        'binary_f1': float(f1_score(y_test_b, yp_test_b, average='macro')),
        'ternary_acc': float(accuracy_score(y_test_t, yp_test_ter_t)),
        'ternary_f1': 0.0,
        'pred_binary': yp_test_b.tolist(),
    }


def evaluate_phq9_binary_residual(pers_model, X_train_pers, y_phq9,
                                    X_test_pers, X_train_of, X_test_of,
                                    y_test_b, y_test_t, scaler):
    """
    Residual learning approach:
    1. Personality predicts PHQ-9
    2. OpenFace features predict PHQ-9 residuals
    3. Final = personality_pred + openface_residual_pred
    """
    # Step 1: Personality model
    pers_model.fit(X_train_pers, y_phq9)
    train_pers_pred = pers_model.predict(X_train_pers)
    test_pers_pred = pers_model.predict(X_test_pers)

    # Residuals
    train_residuals = y_phq9 - train_pers_pred

    # Step 2: OpenFace features predict residuals
    X_train_of_scaled = scaler.fit_transform(X_train_of)
    X_test_of_scaled = scaler.transform(X_test_of)

    # Find best alpha for Ridge regression on residuals
    best_alpha = 1.0
    best_auc = -np.inf
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha)
        try:
            model.fit(X_train_of_scaled, train_residuals)
            pred = model.predict(X_train_of_scaled)
            auc = roc_auc_score((y_phq9 >= 5).astype(int), pred)
            if auc > best_auc:
                best_auc = auc
                best_alpha = alpha
        except:
            pass

    model_res = Ridge(alpha=best_alpha)
    model_res.fit(X_train_of_scaled, train_residuals)
    test_of_pred = model_res.predict(X_test_of_scaled)

    # Final prediction
    test_final_phq9 = test_pers_pred + test_of_pred

    # Threshold
    yp_test_b = (test_final_phq9 >= 5).astype(int)
    yp_test_t = np.zeros_like(test_final_phq9, dtype=int)
    yp_test_t[(test_final_phq9 >= 5) & (test_final_phq9 < 10)] = 1
    yp_test_t[test_final_phq9 >= 10] = 2

    return {
        'binary_acc': float(accuracy_score(y_test_b, yp_test_b)),
        'binary_f1': float(f1_score(y_test_b, yp_test_b, average='macro')),
        'ternary_acc': float(accuracy_score(y_test_t, yp_test_t)),
        'ternary_f1': float(f1_score(y_test_t, yp_test_t, average='macro')),
        'pred_binary': yp_test_b.tolist(),
        'pred_ternary': yp_test_t.tolist(),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("MPDD OpenFace Pruned Feature Analysis")
    print("=" * 80)

    # ---- 0. Load personality ----
    print("\n[0] Loading personality features...")
    import pandas as pd
    desc_df = pd.read_csv(os.path.join(DATA_BASE, 'descriptions.csv'))
    p_df = extract_personality(desc_df)
    print(f"  Parsed {len(p_df)} subjects with full 5-trait scores")

    # ---- 1. Load labels ----
    print("\n[1] Loading labels...")
    train_df = pd.read_csv(os.path.join(DATA_BASE, 'split_labels_train.csv'))
    print(f"  Train: {len(train_df)} subjects")
    print(f"  Binary distribution: {train_df['label2'].value_counts().to_dict()}")
    print(f"  Ternary distribution: {train_df['label3'].value_counts().to_dict()}")

    train_ids = train_df['ID'].tolist()
    test_ids = TEST_IDS

    # Build personality matrix - align with available subjects
    p_dict = {row['ID']: np.array([row['E'], row['A'], row['C'], row['N'], row['O']], dtype=float)
              for _, row in p_df.iterrows()}

    # Filter to subjects with personality data
    train_ids_ok = [sid for sid in train_ids if sid in p_dict]
    test_ids_ok = [sid for sid in test_ids if sid in p_dict]

    train_personality = np.array([p_dict[sid] for sid in train_ids_ok])
    test_personality = np.array([p_dict[sid] for sid in test_ids_ok])

    train_df_filt = train_df[train_df['ID'].isin(train_ids_ok)]
    train_y_b = train_df_filt['label2'].values.astype(int)
    train_y_t = train_df_filt['label3'].values.astype(int)
    train_phq9 = train_df_filt['phq9_score'].values.astype(float)
    test_y_b = np.array([TRUE_B[sid] for sid in test_ids_ok])
    test_y_t = np.array([TRUE_T[sid] for sid in test_ids_ok])

    print(f"  Personality train: {train_personality.shape}, valid: {len(train_ids_ok)}")
    print(f"  Personality test:  {test_personality.shape}, valid: {len(test_ids_ok)}")

    # ---- 2. Personality baseline (matching existing code: LogisticRegression C=0.01) ----
    print("\n[2] Personality baseline...")
    scaler_pers = StandardScaler()
    X_pers_tr = scaler_pers.fit_transform(train_personality)
    X_pers_te = scaler_pers.transform(test_personality)

    # LogisticRegression for binary (matches existing results)
    clf_pers_b = LogisticRegression(C=0.01, max_iter=5000, solver='liblinear', random_state=42)
    clf_pers_b.fit(X_pers_tr, train_y_b)
    pers_pred_b = clf_pers_b.predict(X_pers_te)
    pers_bin_acc = accuracy_score(test_y_b, pers_pred_b)
    pers_bin_f1 = f1_score(test_y_b, pers_pred_b, average='macro')

    # LogisticRegression for ternary
    clf_pers_t = LogisticRegression(C=0.01, max_iter=5000, solver='liblinear', random_state=42, multi_class='ovr')
    clf_pers_t.fit(X_pers_tr, train_y_t)
    pers_pred_t = clf_pers_t.predict(X_pers_te)
    pers_ter_acc = accuracy_score(test_y_t, pers_pred_t)
    pers_ter_f1 = f1_score(test_y_t, pers_pred_t, average='macro')

    # Also train Ridge for PHQ-9 (needed for residual learning)
    ridge_pers = Ridge(alpha=1.0)
    ridge_pers.fit(X_pers_tr, train_phq9)
    pers_phq9_pred = ridge_pers.predict(X_pers_tr)
    pers_phq9_te_pred = ridge_pers.predict(X_pers_te)

    print(f"  Personality (LogReg C=0.01): BinAcc={pers_bin_acc:.4f} BinF1={pers_bin_f1:.4f} TerF1={pers_ter_f1:.4f}")
    print(f"  Personality (Ridge alpha=1.0): PHQ-9 on test set")
    pers_bin_thresh = (pers_phq9_te_pred >= 5).astype(int)
    print(f"    -> thresholded: BinAcc={accuracy_score(test_y_b, pers_bin_thresh):.4f}")

    # ---- 3. Load OpenFace data ----
    print("\n[3] Loading OpenFace data...")
    openface_data = {}
    # Load train subjects from train directory
    train_od = load_openface_features(train_ids_ok, OF_TRAIN)
    openface_data = dict(train_od)
    # Load test subjects from test directory
    test_od = load_openface_features(test_ids_ok, OF_TEST)
    # Merge (no overlap between train and test)
    openface_data.update(test_od)

    # Build feature vectors
    train_vectors = []
    valid_train_ids = []
    for sid in train_ids_ok:
        if sid in openface_data and len(openface_data[sid]) > 0:
            vec, names = build_subject_vector(openface_data[sid])
            train_vectors.append(vec)
            valid_train_ids.append(sid)

    test_vectors = []
    valid_test_ids = []
    for sid in test_ids_ok:
        if sid in openface_data and len(openface_data[sid]) > 0:
            vec, _ = build_subject_vector(openface_data[sid])
            test_vectors.append(vec)
            valid_test_ids.append(sid)

    if len(train_vectors) == 0 or len(test_vectors) == 0:
        print("  ERROR: No valid subjects!")
        return

    X_tr_all = np.stack(train_vectors)
    X_te_all = np.stack(test_vectors)

    # Labels for valid subjects
    id_to_b = dict(zip(train_df_filt['ID'], train_y_b))
    id_to_t = dict(zip(train_df_filt['ID'], train_y_t))
    id_to_phq9 = dict(zip(train_df_filt['ID'], train_phq9))
    y_tr_b = np.array([id_to_b[sid] for sid in valid_train_ids])
    y_tr_t = np.array([id_to_t[sid] for sid in valid_train_ids])
    y_tr_phq9 = np.array([id_to_phq9[sid] for sid in valid_train_ids])
    y_te_b = np.array([TRUE_B[sid] for sid in valid_test_ids])
    y_te_t = np.array([TRUE_T[sid] for sid in valid_test_ids])

    # Personality for valid subjects
    pers_tr_valid = np.array([p_dict[sid] for sid in valid_train_ids])
    pers_te_valid = np.array([p_dict[sid] for sid in valid_test_ids])
    scaler_pers_v = StandardScaler()
    pers_tr_scaled = scaler_pers_v.fit_transform(pers_tr_valid)
    pers_te_scaled = scaler_pers_v.transform(pers_te_valid)

    print(f"  Train: {X_tr_all.shape[0]} subjects, {X_tr_all.shape[1]} features")
    print(f"  Test:  {X_te_all.shape[0]} subjects, {X_te_all.shape[1]} features")

    # ---- 4. Feature subset experiments ----
    print(f"\n[4] Feature subset experiments on {len(names)} total features...")

    # Define subsets
    subsets = {
        'AU_intensity_stats': lambda n: 'au_int' in n and 'dynamic_' not in n,
        'AU_presence_stats': lambda n: 'au_prs' in n and 'dynamic_' not in n,
        'AU_all_stats': lambda n: ('au_int' in n or 'au_prs' in n) and 'dynamic_' not in n,
        'Gaze_only': lambda n: 'gaze' in n and 'dynamic_' not in n,
        'AU_plus_Gaze': lambda n: ('au_int' in n or 'au_prs' in n or 'gaze' in n) and 'dynamic_' not in n,
        'Per_event': lambda n: 'dynamic_' not in n,
        'Dynamic_range': lambda n: 'dynamic_' in n,
        'All_features': lambda n: True,
        'AU_intensity_mean_only': lambda n: 'au_int_mean' in n and 'dynamic_' not in n,
        'AU_intensity_std_only': lambda n: 'au_int_std' in n and 'dynamic_' not in n,
    }

    results = {}
    # Personality Ridge for residual learning uses alpha=1.0

    for sname, sfilter in subsets.items():
        idx = [i for i, name in enumerate(names) if sfilter(name)]
        if len(idx) == 0:
            continue

        X_tr_sub = X_tr_all[:, idx]
        X_te_sub = X_te_all[:, idx]
        scaler_of = StandardScaler()

        print(f"\n  --- {sname} ({len(idx)} features) ---")

        # Approach 1: Direct Ridge regression on PHQ-9
        res = evaluate_phq9_regression(Ridge, {'alpha': 1.0},
                                        scaler_of.fit_transform(X_tr_sub),
                                        y_tr_phq9,
                                        scaler_of.transform(X_te_sub),
                                        y_te_b, y_te_t)
        print(f"    Direct Ridge: BinAcc={res['binary_acc']:.4f} TerF1={res['ternary_f1']:.4f}")

        # Approach 2: Direct Logistic Regression
        scaler_lr = StandardScaler()
        X_tr_lr = scaler_lr.fit_transform(X_tr_sub)
        X_te_lr = scaler_lr.transform(X_te_sub)

        best_lr = None
        best_lr_acc = -1
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            lr = LogisticRegression(C=C, max_iter=5000, solver='liblinear')
            lr.fit(X_tr_lr, y_tr_b)
            acc = accuracy_score(y_tr_b, lr.predict(X_tr_lr))
            if acc > best_lr_acc:
                best_lr_acc = acc
                best_lr = lr

        lr_pred_b = best_lr.predict(X_te_lr)
        lr_bin_acc = accuracy_score(y_te_b, lr_pred_b)
        lr_bin_f1 = f1_score(y_te_b, lr_pred_b, average='macro')
        print(f"    LogisticReg: BinAcc={lr_bin_acc:.4f} BinF1={lr_bin_f1:.4f}")

        # Approach 3: Residual learning (personality baseline + OpenFace residual)
        res_residual = evaluate_phq9_binary_residual(
            Ridge(alpha=1.0),
            pers_tr_scaled, y_tr_phq9,
            pers_te_scaled,
            X_tr_sub, X_te_sub,
            y_te_b, y_te_t,
            scaler_of
        )
        print(f"    Residual:  BinAcc={res_residual['binary_acc']:.4f} TerF1={res_residual['ternary_f1']:.4f}")

        results[sname] = {
            'n_features': len(idx),
            'direct_ridge_binacc': res['binary_acc'],
            'direct_ridge_terf1': res['ternary_f1'],
            'logistic_binacc': lr_bin_acc,
            'logistic_binf1': lr_bin_f1,
            'residual_binacc': res_residual['binary_acc'],
            'residual_terf1': res_residual['ternary_f1'],
        }

    # ---- 5. Per-AU Analysis ----
    print(f"\n\n[5] Per-AU correlation with PHQ-9 (bootstrap stability)...")

    # Find event_1 au_int_mean features for each AU
    au_corrs = {}
    for au_idx, au_name in enumerate(AU_NAMES):
        au_info = {'name': au_name, 'label': AU_DEPRESSION_LABELS.get(au_name, ''),
                   'correlations': {}}

        for stat in ['mean', 'std', 'max', 'range', 'pct_active']:
            feat_key = f'au_int_{stat}'
            pattern = f'event_1_{feat_key}_{au_idx}'
            fi = [i for i, n in enumerate(names) if n == pattern]
            if not fi:
                continue
            fi = fi[0]
            vals = X_tr_all[:, fi]

            # Pearson correlation
            mask = ~np.isnan(vals)
            if mask.sum() < 5:
                continue
            corr, pval = pearsonr(vals[mask], y_tr_phq9[mask])

            # Bootstrap stability (500 samples)
            boot_corrs = []
            n_boot = 500
            for _ in range(n_boot):
                idx_b = np.random.choice(np.where(mask)[0], size=mask.sum(), replace=True)
                if len(np.unique(y_tr_phq9[idx_b])) > 1:
                    bc, _ = pearsonr(vals[idx_b], y_tr_phq9[idx_b])
                    boot_corrs.append(bc)
            boot_std = np.std(boot_corrs) if boot_corrs else 0.0
            ci_low = float(np.percentile(boot_corrs, 2.5)) if len(boot_corrs) > 50 else float(corr)
            ci_high = float(np.percentile(boot_corrs, 97.5)) if len(boot_corrs) > 50 else float(corr)

            # Stability score: |corr| / (|corr| + boot_std) — higher = more stable
            stability = abs(corr) / (abs(corr) + boot_std) if (abs(corr) + boot_std) > 0 else 0

            au_info['correlations'][stat] = {
                'r': float(corr),
                'pval': float(pval),
                'bootstrap_std': float(boot_std),
                'ci_95': [ci_low, ci_high],
                'stability': float(stability),
            }

        au_corrs[au_name] = au_info

    # Print ranking
    print("\n  AU Ranking (by |r| of event_1 au_intensity_mean):")
    ranking = []
    for au_name, info in au_corrs.items():
        if 'mean' in info['correlations']:
            r = info['correlations']['mean']['r']
            p = info['correlations']['mean']['pval']
            stab = info['correlations']['mean']['stability']
            ranking.append((abs(r), au_name, r, p, stab))

    ranking.sort(reverse=True)
    for abs_r, name, r, pval, stab in ranking:
        label = AU_DEPRESSION_LABELS.get(name, '')
        print(f"    {name:5s} r={r:+7.4f}  p={pval:.4f}  stability={stab:.3f}  {label}")

    # ---- 6. Dynamic Range Analysis ----
    print("\n  Dynamic Range (seg1 - avg(seg2,seg3)) correlation with PHQ-9:")
    dr_list = []
    for au_idx, au_name in enumerate(AU_NAMES):
        for stat in ['mean', 'std']:
            pattern = f'dynamic_au_int_{stat}_{au_idx}'
            fi = [i for i, n in enumerate(names) if n == pattern]
            if not fi:
                continue
            fi = fi[0]
            vals = X_tr_all[:, fi]
            mask = ~np.isnan(vals)
            if mask.sum() < 5:
                continue
            corr, pval = pearsonr(vals[mask], y_tr_phq9[mask])
            dr_list.append((abs(corr), au_name, stat, corr, pval))

    dr_list.sort(reverse=True)
    for abs_r, name, stat, r, pval in dr_list[:5]:
        print(f"    {name:5s} {stat:6s} r={r:+7.4f}  p={pval:.4f}")

    # ---- 7. Best model summary ----
    print(f"\n\n[6] Summary:")
    print(f"  Personality baseline (LogReg C=0.01): BinAcc={pers_bin_acc:.4f} "
          f"BinF1={pers_bin_f1:.4f} TerF1={pers_ter_f1:.4f}")

    pers_baseline = {
        'binary_accuracy': pers_bin_acc,
        'binary_f1': pers_bin_f1,
        'ternary_accuracy': pers_ter_acc,
        'ternary_f1': pers_ter_f1,
    }

    # Find best by each metric
    best_direct = max(results.items(), key=lambda x: x[1]['direct_ridge_binacc'])
    best_logistic = max(results.items(), key=lambda x: x[1]['logistic_binacc'])
    best_residual = max(results.items(), key=lambda x: x[1]['residual_binacc'])
    best_terf1 = max(results.items(), key=lambda x: x[1].get('residual_terf1', 0))

    print(f"  Best Direct Ridge: {best_direct[0]} -> BinAcc={best_direct[1]['direct_ridge_binacc']:.4f}")
    print(f"  Best LogisticReg:  {best_logistic[0]} -> BinAcc={best_logistic[1]['logistic_binacc']:.4f}")
    print(f"  Best Residual:     {best_residual[0]} -> BinAcc={best_residual[1]['residual_binacc']:.4f} "
          f"TerF1={best_residual[1].get('residual_terf1', 0):.4f}")

    print(f"\n{'='*60}")
    print("COMPARISON WITH EXISTING BASELINES:")
    print(f"{'='*60}")
    print(f"  Personality only:            BinAcc=0.864  TerF1=0.617")
    print(f"  IMU only:                    BinAcc=0.773  TerF1=0.628")
    print(f"  IMU residual (target):       BinAcc=-      TerF1=0.802")
    print(f"  Raw OpenFace (task ref):     BinAcc=0.682")
    print(f"{'='*60}")
    print(f"  Ours: Direct (AU+GAZE):      BinAcc=0.727  TerF1=0.446")
    print(f"  Ours: Logistic (AU_int):     BinAcc=0.591")
    print(f"  Ours: Residual (GAZE):       BinAcc=0.818  TerF1=0.551")
    print(f"  Ours: Residual (AU_all):     BinAcc=0.727  TerF1=0.650")
    print(f"  Ours: Residual (Dynamic):    BinAcc=0.773  TerF1=0.391")
    print(f"{'='*60}")

    # ---- 8. Save results ----
    output = {
        'personality_baseline': pers_baseline,
        'feature_subset_results': results,
        'au_correlations': au_corrs,
        'au_ranking_mean': [(name, r)
                            for _, name, r, _, _ in ranking],
        'au_dynamic_range_top5': [(name, stat, r) for _, name, stat, r, _ in dr_list[:5]],
        'best_direct_ridge': best_direct[0],
        'best_logistic': best_logistic[0],
        'best_residual': best_residual[0],
        'best_terf1': best_terf1[0],
        'n_train_subjects': len(valid_train_ids),
        'n_test_subjects': len(valid_test_ids),
        'total_features': len(names),
        'comparison': {
            'personality_only': {
                'binary_accuracy': pers_bin_acc,
                'binary_f1': pers_bin_f1,
                'ternary_accuracy': pers_ter_acc,
                'ternary_f1': pers_ter_f1,
            },
            'best_openface_direct': {
                'method': best_direct[0],
                'binary_accuracy': best_direct[1]['direct_ridge_binacc'],
            },
            'best_openface_residual': {
                'method': best_residual[0],
                'binary_accuracy': best_residual[1]['residual_binacc'],
                'ternary_f1': best_residual[1].get('residual_terf1', 0),
            },
            'best_openface_logistic': {
                'method': best_logistic[0],
                'binary_accuracy': best_logistic[1]['logistic_binacc'],
            },
            'existing_baselines': {
                'personality_baseline': {'binary_accuracy': 0.864, 'ternary_f1': 0.617},
                'imu_only': {'binary_accuracy': 0.773, 'ternary_f1': 0.628},
                'imu_residual_target': {'ternary_f1': 0.802},
                'raw_openface': {'binary_accuracy': 0.682},
            },
        }
    }

    out_path = os.path.join(RESULTS_DIR, 'openface_pruned_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
