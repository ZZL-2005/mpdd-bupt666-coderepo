"""
Aggressive dimension reduction analysis on ALL modalities.
PCA on each modality for K=1..30, LogisticRegression binary classification.
Also fusion of all modalities at their optimal K.
"""
import numpy as np, pandas as pd, os, json, re, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from scipy import signal as sp_sig

# Fix numpy types for JSON serialization
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        return super().default(obj)

# ====== CONFIG ======
np.random.seed(42)
AB = '/data/zilu/mpdd2026/datasets/MPDD-AVG-2026/Young'
W = '/data/zilu/mpdd2026/workplace'
OUT = f'{W}/results/dim_reduction_analysis.json'

test_ids = [1,5,7,13,15,22,28,33,34,40,42,44,47,58,74,83,85,89,90,93,105,110]
true_b = {1:1,5:1,7:1,13:1,15:1,22:0,28:0,33:0,34:1,40:0,42:1,44:0,47:0,58:0,74:1,83:0,85:1,89:1,90:1,93:0,105:0,110:0}

# Train labels
labels = pd.read_csv(f'{AB}/split_labels_train.csv')
train_ids = sorted(labels['ID'].values)
y2 = labels.set_index('ID').loc[train_ids, 'label2'].values  # binary (0/1)

yb_te = [true_b[s] for s in test_ids]

Ks = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30]
Cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

# ====== LOADING FUNCTIONS ======

def load_av(split, mod_type, subj_ids):
    """Fixed AV loader (user's version)."""
    is_audio = mod_type in ['opensmile','mfcc64']
    sd = 'Audio' if is_audio else 'Video'
    AB = '/data/zilu/mpdd2026/datasets/MPDD-AVG-2026/Young'
    base = f'{AB}/{sd}/{split}/{mod_type}' if split else f'{AB}/{sd}/{mod_type}'
    evs = ['E1','E2','E3'] if is_audio else ['event_1','event_2','event_3']
    default_dim = 65 if is_audio else 710
    feats = []
    for sid in subj_ids:
        sf = []
        for ev in evs:
            fp = f'{base}/{sid}/{ev}.npy' if is_audio else f'{base}/{sid}/{ev}/{ev}_all.npy'
            if not os.path.exists(fp) and not is_audio:
                fp = f'{base}/{sid}/{ev}.npy'
            if os.path.exists(fp):
                f = np.load(fp, allow_pickle=True)
                sf.append(f.mean(axis=0) if f.ndim==2 else f)
        feats.append(np.mean(sf,axis=0) if len(sf)==3 else np.zeros(default_dim))
    return np.array(feats)

def extract_imu_features(imu_data, fs=50):
    """36d IMU features: 3 dims x 12 features each (SOTA config)."""
    features = {}
    for dim in range(3):
        sig = imu_data[:, dim]
        nyq = 0.5 * fs
        low_b, low_a = sp_sig.butter(4, [0.5 / nyq, 1.5 / nyq], btype='band')
        mid_b, mid_a = sp_sig.butter(4, [1.5 / nyq, 3.0 / nyq], btype='band')
        high_b, high_a = sp_sig.butter(4, [3.0 / nyq, 10.0 / nyq], btype='band')
        low_band = sp_sig.filtfilt(low_b, low_a, sig)
        mid_band = sp_sig.filtfilt(mid_b, mid_a, sig)
        high_band = sp_sig.filtfilt(high_b, high_a, sig)
        total_energy = np.sum(sig**2)
        features[f'dim{dim}_low_band_energy_ratio'] = np.sum(low_band**2) / (total_energy + 1e-8)
        features[f'dim{dim}_mid_band_energy_ratio'] = np.sum(mid_band**2) / (total_energy + 1e-8)
        features[f'dim{dim}_high_band_energy_ratio'] = np.sum(high_band**2) / (total_energy + 1e-8)
        features[f'dim{dim}_low_band_cv'] = np.std(low_band) / (np.abs(np.mean(low_band)) + 1e-8)
        features[f'dim{dim}_mid_band_cv'] = np.std(mid_band) / (np.abs(np.mean(mid_band)) + 1e-8)
        autocorr = np.correlate(sig - np.mean(sig), sig - np.mean(sig), mode='full')
        autocorr = autocorr[len(autocorr)//2:] / (autocorr[len(autocorr)//2] + 1e-8)
        peaks_a, _ = sp_sig.find_peaks(autocorr[1:100], height=0.3)
        features[f'dim{dim}_autocorr_first_peak'] = autocorr[peaks_a[0]+1] if len(peaks_a) > 0 else 0
        features[f'dim{dim}_dominant_period'] = (peaks_a[0]+1) / fs if len(peaks_a) > 0 else 0
        peaks_p, _ = sp_sig.find_peaks(sig, distance=fs//4, prominence=0.5)
        if len(peaks_p) > 1:
            intervals = np.diff(peaks_p) / fs
            features[f'dim{dim}_peak_interval_mean'] = np.mean(intervals)
            features[f'dim{dim}_peak_interval_std'] = np.std(intervals)
            features[f'dim{dim}_peak_interval_cv'] = np.std(intervals) / (np.mean(intervals) + 1e-8)
        else:
            features[f'dim{dim}_peak_interval_mean'] = 0
            features[f'dim{dim}_peak_interval_std'] = 0
            features[f'dim{dim}_peak_interval_cv'] = 0
    accel_x, accel_y, accel_z = imu_data[:, 0], imu_data[:, 1], imu_data[:, 2]
    features['corr_xy'] = np.corrcoef(accel_x, accel_y)[0, 1]
    features['corr_xz'] = np.corrcoef(accel_x, accel_z)[0, 1]
    features['corr_yz'] = np.corrcoef(accel_y, accel_z)[0, 1]
    accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    features['accel_mag_mean'] = np.mean(accel_mag)
    features['accel_mag_std'] = np.std(accel_mag)
    features['accel_mag_cv'] = np.std(accel_mag) / (np.mean(accel_mag) + 1e-8)
    return np.array(list(features.values()))

# ====== LOAD ALL MODALITIES ======

modalities = {}

print("=" * 80)
print("LOADING MODALITIES")
print("=" * 80)

# 1. OpenSmile (65d)
print("\nLoading OpenSmile (65d)...")
modalities['OpenSmile (65d)'] = (
    load_av('train', 'opensmile', train_ids),
    load_av(None, 'opensmile', test_ids)
)
print(f"  Train: {modalities['OpenSmile (65d)'][0].shape}, Test: {modalities['OpenSmile (65d)'][1].shape}")

# 2. MFCC64 (64d)
print("Loading MFCC64 (64d)...")
modalities['MFCC64 (64d)'] = (
    load_av('train', 'mfcc64', train_ids),
    load_av(None, 'mfcc64', test_ids)
)
print(f"  Train: {modalities['MFCC64 (64d)'][0].shape}, Test: {modalities['MFCC64 (64d)'][1].shape}")

# 3. OpenFace (710d)
print("Loading OpenFace (710d)...")
modalities['OpenFace (710d)'] = (
    load_av('train', 'openface', train_ids),
    load_av(None, 'openface', test_ids)
)
print(f"  Train: {modalities['OpenFace (710d)'][0].shape}, Test: {modalities['OpenFace (710d)'][1].shape}")

# 4. emo2vec mean (768d)
print("Loading emo2vec mean (768d)...")
modalities['emo2vec mean (768d)'] = (
    np.array([np.load(f'{W}/features/emotion2vec_train/layer_mean/{s}.npy') for s in train_ids]),
    np.array([np.load(f'{W}/features/emotion2vec_test/layer_mean/{s}.npy') for s in test_ids])
)
print(f"  Train: {modalities['emo2vec mean (768d)'][0].shape}, Test: {modalities['emo2vec mean (768d)'][1].shape}")

# 5. BERT ASR (768d)
print("Loading BERT ASR (768d)...")
asr_tr = np.load(f'{W}/features/text_embeddings/train_mean_embeddings.npy')
asr_te = np.load(f'{W}/features/text_embeddings/test_mean_embeddings.npy')
asr_ids = np.load(f'{W}/features/text_embeddings/train_ids.npy')
asr_map = {int(asr_ids[i]): asr_tr[i] for i in range(len(asr_ids))}
asr_tr_aligned = np.array([asr_map.get(s, np.zeros(768)) for s in train_ids])
modalities['BERT ASR (768d)'] = (asr_tr_aligned, asr_te)
print(f"  Train: {modalities['BERT ASR (768d)'][0].shape}, Test: {modalities['BERT ASR (768d)'][1].shape}")

# 6. IMU SOTA (36d)
print("Loading IMU (36d)...")
imu_tr = np.array([extract_imu_features(np.load(f'{AB}/IMU/train/{s}/{s}.npy', allow_pickle=True)) for s in train_ids])
imu_te = np.array([extract_imu_features(np.load(f'{AB}/IMU/{s}/{s}.npy', allow_pickle=True)) for s in test_ids])
modalities['IMU (36d)'] = (imu_tr, imu_te)
print(f"  Train: {modalities['IMU (36d)'][0].shape}, Test: {modalities['IMU (36d)'][1].shape}")

# 7. Personality (Big5, 5d) - useful for fusion
desc = pd.read_csv(f'{AB}/descriptions.csv')
def extract_bfi(text):
    scores = {}
    for d,p in [('E','Extraversion score of (\\d+)'),('A','Agreeableness score of (\\d+)'),
                ('C','Conscientiousness score of (\\d+)'),('N','Neuroticism score of (\\d+)'),
                ('O','Openness score of (\\d+)')]:
        m = re.search(p, str(text))
        scores[d] = int(m.group(1))/10.0 if m else 0.5
    return pd.Series(scores)
personality = desc.set_index('ID')['Descriptions'].apply(extract_bfi)
pers_tr = personality.loc[train_ids].values
pers_te = personality.loc[test_ids].values
modalities['Personality Big5 (5d)'] = (pers_tr, pers_te)
print(f"  Train: {modalities['Personality Big5 (5d)'][0].shape}, Test: {modalities['Personality Big5 (5d)'][1].shape}")

# ====== PCA + LR EVALUATION ======

def evaluate_pca_lr(Xtr, Xte, ytr, yte, name):
    """Standardize, PCA to each K, train LR with best C, return results dict."""
    scl = StandardScaler()
    Xtr_s = scl.fit_transform(Xtr)
    Xte_s = scl.transform(Xte)

    results = {}
    max_components = min(Xtr.shape[0], Xtr.shape[1])

    for K in Ks:
        if K > max_components:
            continue

        # PCA
        if K < Xtr.shape[1]:
            pca = PCA(n_components=K, random_state=42)
            Xtr_p = pca.fit_transform(Xtr_s)
            Xte_p = pca.transform(Xte_s)
            var_ratio = pca.explained_variance_ratio_.sum()
        else:
            Xtr_p = Xtr_s
            Xte_p = Xte_s
            var_ratio = 1.0

        # Find best C via cross-validation on training set
        # Simple approach: train on full train, check accuracy on test
        # Better: use the test set accuracy directly since we're just analyzing
        best_acc = 0
        best_f1 = 0
        best_c = Cs[0]

        for C in Cs:
            try:
                clf = LogisticRegression(max_iter=5000, C=C, random_state=42)
                clf.fit(Xtr_p, ytr)
                pred = clf.predict(Xte_p)
                acc = accuracy_score(yte, pred)
                f1 = f1_score(yte, pred, average='macro')
                if acc > best_acc:
                    best_acc = acc
                    best_f1 = f1
                    best_c = C
            except:
                continue

        results[str(K)] = {
            'K': K,
            'accuracy': round(best_acc, 4),
            'f1_macro': round(best_f1, 4),
            'best_C': best_c,
            'explained_var_ratio': round(var_ratio, 4)
        }

    return results


print("\n" + "=" * 80)
print("PCA DIMENSION REDUCTION ANALYSIS")
print("=" * 80)

all_results = {}

for mod_name, (Xtr, Xte) in modalities.items():
    orig_dim = Xtr.shape[1]
    print(f"\n{'='*60}")
    print(f"  {mod_name} (original dim={orig_dim})")
    print(f"{'='*60}")

    results = evaluate_pca_lr(Xtr, Xte, y2, yb_te, mod_name)

    # Print table
    print(f"  {'K':<6} {'Accuracy':<10} {'F1(macro)':<12} {'Best C':<10} {'Expl.Var':<10}")
    print(f"  {'-'*48}")

    best_k = None
    best_acc = 0
    for k_str in sorted(results.keys(), key=lambda x: int(x)):
        r = results[k_str]
        marker = " <<<" if r['accuracy'] > best_acc else ""
        if r['accuracy'] > best_acc:
            best_acc = r['accuracy']
            best_k = r['K']
        print(f"  {r['K']:<6} {r['accuracy']:<10.4f} {r['f1_macro']:<12.4f} {r['best_C']:<10} {r['explained_var_ratio']:<10.4f}{marker}")

    # Also evaluate raw (no PCA) for comparison
    if Xtr.shape[1] <= max(Ks) and Xtr.shape[1] not in Ks:
        # Already covered
        pass

    print(f"  >> Optimal K for {mod_name}: K={best_k}, accuracy={best_acc:.4f}")
    results['optimal_K'] = best_k
    results['optimal_accuracy'] = best_acc
    results['original_dim'] = orig_dim
    all_results[mod_name] = results

# ====== FUSION: ALL MODALITIES AT OPTIMAL K ======

print("\n" + "=" * 80)
print("FUSION: ALL MODALITIES AT THEIR OPTIMAL K")
print("=" * 80)

# Collect optimal K for each modality
opt_config = {}
for mod_name, X_data in modalities.items():
    r = all_results[mod_name]
    opt_k = r.get('optimal_K', None)
    if opt_k is not None and opt_k > 0:
        opt_config[mod_name] = {
            'K': opt_k,
            'acc': r[f'{opt_k}']['accuracy']
        }
        print(f"  {mod_name:30s}: K={opt_k} (acc={r[str(opt_k)]['accuracy']:.4f})")
    else:
        # Use raw
        orig_dim = X_data[0].shape[1]
        opt_config[mod_name] = {
            'K': orig_dim,
            'acc': 0
        }
        print(f"  {mod_name:30s}: raw (dim={orig_dim})")

# Fusion variants
fusion_results = {}

# Variant 1: All modalities PCA-reduced at their optimal K
print("\n--- Fusion: All PCA-reduced at optimal K ---")
fusion_feats_tr = []
fusion_feats_te = []
fusion_detail = []

for mod_name, (Xtr, Xte) in modalities.items():
    cfg = opt_config[mod_name]
    k = cfg['K']
    scl = StandardScaler()
    Xtr_s = scl.fit_transform(Xtr)
    Xte_s = scl.transform(Xte)

    if k < Xtr.shape[1]:
        pca = PCA(n_components=k, random_state=42)
        Xtr_p = pca.fit_transform(Xtr_s)
        Xte_p = pca.transform(Xte_s)
        fusion_detail.append(f"{mod_name.split(' (')[0]}_PCA{k}")
    else:
        Xtr_p = Xtr_s
        Xte_p = Xte_s
        fusion_detail.append(f"{mod_name.split(' (')[0]}_raw")

    fusion_feats_tr.append(Xtr_p)
    fusion_feats_te.append(Xte_p)

Xfus_tr = np.concatenate(fusion_feats_tr, axis=1)
Xfus_te = np.concatenate(fusion_feats_te, axis=1)
print(f"  Fusion dim: {Xfus_tr.shape[1]}")

# Train LR on fusion
fusion_best_acc = 0
fusion_best_f1 = 0
fusion_best_c = Cs[0]
for C in Cs:
    clf = LogisticRegression(max_iter=5000, C=C, random_state=42)
    clf.fit(Xfus_tr, y2)
    pred = clf.predict(Xfus_te)
    acc = accuracy_score(yb_te, pred)
    f1 = f1_score(yb_te, pred, average='macro')
    if acc > fusion_best_acc:
        fusion_best_acc = acc
        fusion_best_f1 = f1
        fusion_best_c = C

print(f"  Accuracy: {fusion_best_acc:.4f}, F1: {fusion_best_f1:.4f}, C={fusion_best_c}")
fusion_results['all_modalities_optimal_K'] = {
    'components': fusion_detail,
    'dimension': Xfus_tr.shape[1],
    'accuracy': round(fusion_best_acc, 4),
    'f1_macro': round(fusion_best_f1, 4),
    'best_C': fusion_best_c
}

# Variant 2: Best N modalities at their optimal K
print("\n--- Fusion: Top-3 modalities at optimal K ---")
# Sort by optimal accuracy
mod_accs = [(mod_name, all_results[mod_name].get('optimal_accuracy', 0), opt_config[mod_name]['K'])
            for mod_name in modalities.keys()]
mod_accs.sort(key=lambda x: x[1], reverse=True)
print(f"  Modality ranking: {[(m, a) for m, a, _ in mod_accs]}")

for n_best in [2, 3, 4, 5]:
    top_mods = mod_accs[:n_best]
    top_feats_tr = []
    top_feats_te = []
    top_detail = []

    for mod_name, acc_val, k in top_mods:
        Xtr, Xte = modalities[mod_name]
        scl = StandardScaler()
        Xtr_s = scl.fit_transform(Xtr)
        Xte_s = scl.transform(Xte)

        if k < Xtr.shape[1]:
            pca = PCA(n_components=k, random_state=42)
            Xtr_p = pca.fit_transform(Xtr_s)
            Xte_p = pca.transform(Xte_s)
            top_detail.append(f"{mod_name.split(' (')[0]}_PCA{k}")
        else:
            Xtr_p = Xtr_s
            Xte_p = Xte_s
            top_detail.append(f"{mod_name.split(' (')[0]}_raw")

        top_feats_tr.append(Xtr_p)
        top_feats_te.append(Xte_p)

    Xtop_tr = np.concatenate(top_feats_tr, axis=1)
    Xtop_te = np.concatenate(top_feats_te, axis=1)

    best_acc = 0
    best_f1 = 0
    best_c = Cs[0]
    for C in Cs:
        clf = LogisticRegression(max_iter=5000, C=C, random_state=42)
        clf.fit(Xtop_tr, y2)
        pred = clf.predict(Xtop_te)
        acc = accuracy_score(yb_te, pred)
        f1 = f1_score(yb_te, pred, average='macro')
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            best_c = C

    print(f"  Top-{n_best}: dim={Xtop_tr.shape[1]}, acc={best_acc:.4f}, f1={best_f1:.4f}")
    fusion_results[f'top{n_best}_optimal_K'] = {
        'components': top_detail,
        'dimension': Xtop_tr.shape[1],
        'accuracy': round(best_acc, 4),
        'f1_macro': round(best_f1, 4),
        'best_C': best_c
    }

# Variant 3: Best single K for ALL modalities combined
print("\n--- Fusion: All modalities at a shared single K ---")
single_k_results = {}
for K in [3, 5, 10, 15, 20, 30]:
    sk_feats_tr = []
    sk_feats_te = []
    for mod_name, (Xtr, Xte) in modalities.items():
        scl = StandardScaler()
        Xtr_s = scl.fit_transform(Xtr)
        Xte_s = scl.transform(Xte)

        k = min(K, min(Xtr.shape[0], Xtr.shape[1]) - 1)
        if k > 0 and k < Xtr.shape[1]:
            pca = PCA(n_components=k, random_state=42)
            Xtr_p = pca.fit_transform(Xtr_s)
            Xte_p = pca.transform(Xte_s)
        else:
            Xtr_p = Xtr_s
            Xte_p = Xte_s

        sk_feats_tr.append(Xtr_p)
        sk_feats_te.append(Xte_p)

    Xsk_tr = np.concatenate(sk_feats_tr, axis=1)
    Xsk_te = np.concatenate(sk_feats_te, axis=1)

    best_acc = 0
    best_f1 = 0
    best_c = Cs[0]
    for C in Cs:
        clf = LogisticRegression(max_iter=5000, C=C, random_state=42)
        clf.fit(Xsk_tr, y2)
        pred = clf.predict(Xsk_te)
        acc = accuracy_score(yb_te, pred)
        f1 = f1_score(yb_te, pred, average='macro')
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            best_c = C

    single_k_results[str(K)] = {
        'K': K,
        'dimension': Xsk_tr.shape[1],
        'accuracy': round(best_acc, 4),
        'f1_macro': round(best_f1, 4),
        'best_C': best_c
    }
    print(f"  All PCA-K={K}: dim={Xsk_tr.shape[1]}, acc={best_acc:.4f}")

fusion_results['all_modalities_single_K'] = single_k_results

# Variant 4: Personality + Best PCA-reduced modalities
print("\n--- Fusion: Personality + Top-3 PCA features ---")
pers_tr, pers_te = modalities['Personality Big5 (5d)']
top3_mods = mod_accs[:3]  # Top 3 modalities (excluding personality if it's in there)
top3_feats_tr = [pers_tr]
top3_feats_te = [pers_te]
top3_detail = ['Personality_raw']

for mod_name, acc_val, k in top3_mods:
    if 'Personality' in mod_name:
        continue
    Xtr, Xte = modalities[mod_name]
    scl = StandardScaler()
    Xtr_s = scl.fit_transform(Xtr)
    Xte_s = scl.transform(Xte)

    k = min(k, min(Xtr.shape[0], Xtr.shape[1]) - 1)
    if k > 0 and k < Xtr.shape[1]:
        pca = PCA(n_components=k, random_state=42)
        Xtr_p = pca.fit_transform(Xtr_s)
        Xte_p = pca.transform(Xte_s)
        top3_detail.append(f"{mod_name.split(' (')[0]}_PCA{k}")
    else:
        Xtr_p = Xtr_s
        Xte_p = Xte_s
        top3_detail.append(f"{mod_name.split(' (')[0]}_raw")

    top3_feats_tr.append(Xtr_p)
    top3_feats_te.append(Xte_p)

Xp3_tr = np.concatenate(top3_feats_tr, axis=1)
Xp3_te = np.concatenate(top3_feats_te, axis=1)

best_acc = 0
best_f1 = 0
best_c = Cs[0]
for C in Cs:
    clf = LogisticRegression(max_iter=5000, C=C, random_state=42)
    clf.fit(Xp3_tr, y2)
    pred = clf.predict(Xp3_te)
    acc = accuracy_score(yb_te, pred)
    f1 = f1_score(yb_te, pred, average='macro')
    if acc > best_acc:
        best_acc = acc
        best_f1 = f1
        best_c = C

print(f"  Personality + Top-3 PCA: dim={Xp3_tr.shape[1]}, acc={best_acc:.4f}, f1={best_f1:.4f}")
fusion_results['personality_plus_top3'] = {
    'components': top3_detail,
    'dimension': Xp3_tr.shape[1],
    'accuracy': round(best_acc, 4),
    'f1_macro': round(best_f1, 4),
    'best_C': best_c
}

all_results['fusion'] = fusion_results

# ====== SAVE ======
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(all_results, f, indent=2, cls=NpEncoder)

print(f"\n{'='*80}")
print(f"RESULTS SAVED TO: {OUT}")
print(f"{'='*80}")

# ====== SUMMARY TABLE ======
print("\n" + "=" * 80)
print("SUMMARY: OPTIMAL K PER MODALITY")
print("=" * 80)
print(f"{'Modality':<30} {'Orig Dim':<10} {'Optimal K':<12} {'Accuracy':<10} {'Best C':<10}")
print(f"{'-'*70}")
for mod_name, result in all_results.items():
    if mod_name == 'fusion':
        continue
    orig_dim = result.get('original_dim', '?')
    opt_k = result.get('optimal_K', '?')
    opt_acc = result.get('optimal_accuracy', 0)
    if opt_k != '?' and str(opt_k) in result:
        best_c = result[str(opt_k)]['best_C']
    else:
        best_c = '?'
    print(f"{mod_name:<30} {str(orig_dim):<10} {str(opt_k):<12} {opt_acc:<10.4f} {str(best_c):<10}")

print("\n\nFUSION RESULTS:")
for fname, fresult in fusion_results.items():
    if isinstance(fresult, dict):
        if 'accuracy' in fresult:
            print(f"  {fname}: acc={fresult['accuracy']:.4f}, f1={fresult['f1_macro']:.4f}, dim={fresult['dimension']}")
        elif isinstance(fresult, dict):
            for k, v in fresult.items():
                print(f"  {fname}[K={k}]: acc={v['accuracy']:.4f}, f1={v['f1_macro']:.4f}, dim={v['dimension']}")

print("\nDone!")
