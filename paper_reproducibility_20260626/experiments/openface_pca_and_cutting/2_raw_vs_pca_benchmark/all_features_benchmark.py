"""Systematic benchmark of ALL features for binary classification."""
import numpy as np, pandas as pd, re, os, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from scipy import signal as sp_sig

test_ids = [1,5,7,13,15,22,28,33,34,40,42,44,47,58,74,83,85,89,90,93,105,110]
true_b = {1:1,5:1,7:1,13:1,15:1,22:0,28:0,33:0,34:1,40:0,42:1,44:0,47:0,58:0,74:1,83:0,85:1,89:1,90:1,93:0,105:0,110:0}
true_t = {1:2,5:1,7:1,13:1,15:1,22:0,28:0,33:0,34:1,40:0,42:1,44:0,47:0,58:0,74:2,83:0,85:2,89:1,90:1,93:0,105:0,110:0}

AB = '/data/zilu/mpdd2026/datasets/MPDD-AVG-2026/Young'
labels = pd.read_csv(f'{AB}/split_labels_train.csv'); train_ids = sorted(labels['ID'].values)
lm = labels.set_index('ID')
y2 = lm.loc[train_ids, 'label2'].values; y3 = lm.loc[train_ids, 'label3'].values
yb_te = [true_b[s] for s in test_ids]; yt_te = [true_t[s] for s in test_ids]
W = '/data/zilu/mpdd2026/workplace'

# Helpers
def best_lr_binary(Xtr, Xte, name, max_pca=29):
    """Try multiple C and PCA, return best binary accuracy."""
    scl = StandardScaler()
    Xtr_s = scl.fit_transform(Xtr); Xte_s = scl.transform(Xte)

    # Try without PCA if dims are reasonable
    best = (0, 0, 0, 'none')
    for n_pca in [0, 5, 10, 20, 29]:
        if n_pca > 0 and Xtr.shape[1] > 20:
            if n_pca > min(Xtr.shape[0], Xtr.shape[1]): continue
            pca = PCA(n_pca)
            Xp_tr = pca.fit_transform(Xtr_s); Xp_te = pca.transform(Xte_s)
        elif n_pca > 0 and Xtr.shape[1] <= 20:
            continue  # no need for PCA
        else:
            Xp_tr, Xp_te = Xtr_s, Xte_s

        for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            try:
                c2 = LogisticRegression(max_iter=3000, C=C, random_state=42).fit(Xp_tr, y2)
                bp = c2.predict(Xp_te); ba = accuracy_score(yb_te, bp); bf = f1_score(yb_te, bp, average='macro')
                if ba > best[0]: best = (ba, bf, C, f'PCA{n_pca}' if n_pca>0 else 'raw')
            except: pass
    return best

# ====== FEATURE SETS ======
features = {}

# 1. Personality (Big Five)
desc = pd.read_csv(f'{AB}/descriptions.csv')
def extract_bfi(text):
    scores = {}
    for d,p in [('E','Extraversion score of (\\d+)'),('A','Agreeableness score of (\\d+)'),
                ('C','Conscientiousness score of (\\d+)'),('N','Neuroticism score of (\\d+)'),
                ('O','Openness score of (\\d+)')]:
        m = re.search(p, str(text)); scores[d] = int(m.group(1))/10.0 if m else 0.5
    return pd.Series(scores)
personality = desc.set_index('ID')['Descriptions'].apply(extract_bfi)
features['Personality (5d)'] = (personality.loc[train_ids].values, personality.loc[test_ids].values)

# 2. Demographics
def extract_demo(text):
    age = int(re.search(r'(\d+)-year-old', str(text)).group(1))/20.0 if re.search(r'(\d+)-year-old', str(text)) else 0.9
    gender = 1.0 if 'female' in str(text) else 0.0
    return np.array([age, gender, age*gender])
demo = desc.set_index('ID')['Descriptions'].apply(extract_demo)
features['Demographics (3d)'] = (np.array([demo[s] for s in train_ids]), np.array([demo[s] for s in test_ids]))

# 3. Big5 + interaction terms
pers_vals = personality.loc[train_ids].values
pers_test = personality.loc[test_ids].values
N_train = pers_vals[:, 3]  # Neuroticism
E_train = pers_vals[:, 0]  # Extraversion
N_test = pers_test[:, 3]; E_test = pers_test[:, 0]
demo_train = np.array([demo[s] for s in train_ids])
demo_test = np.array([demo[s] for s in test_ids])
interact_train = np.column_stack([N_train * demo_train[:,1], E_train * demo_train[:,2]])
interact_test = np.column_stack([N_test * demo_test[:,1], E_test * demo_test[:,2]])
features['Big5 + Interactions (7d)'] = (
    np.column_stack([pers_vals, interact_train]),
    np.column_stack([pers_test, interact_test]))

# 4. IMU
imu_dir = f'{AB}/IMU'
def extract_imu(imu_data):
    feats = []
    for dim in range(3):
        sig = imu_data[:, dim]
        low = sp_sig.filtfilt(*sp_sig.butter(4, [0.5/25, 1.5/25], btype='band'), sig) if len(sig)>8 else sig
        mid = sp_sig.filtfilt(*sp_sig.butter(4, [1.5/25, 3.0/25], btype='band'), sig) if len(sig)>8 else sig
        high = sp_sig.filtfilt(*sp_sig.butter(4, [3.0/25, 10.0/25], btype='band'), sig) if len(sig)>8 else sig
        t = np.sum(low**2)+np.sum(mid**2)+np.sum(high**2)+1e-10
        feats.extend([np.sum(low**2)/t, np.sum(mid**2)/t, np.sum(high**2)/t])
        feats.extend([np.std(low)/(np.std(mid)+1e-10), np.std(high)/(np.std(mid)+1e-10)])
        ac = np.correlate(sig-np.mean(sig), sig-np.mean(sig), mode='full'); ac=ac[len(ac)//2:]; ac/=(ac[0]+1e-10)
        p=sp_sig.find_peaks(ac[1:])[0]; feats.extend([ac[p[0]+1] if len(p)>0 else 0, p[0] if len(p)>0 else 0])
        p2=sp_sig.find_peaks(sig)[0]
        if len(p2)>1: iv=np.diff(p2); feats.extend([np.mean(iv),np.std(iv),np.std(iv)/(np.mean(iv)+1e-10)])
        else: feats.extend([0,0,0])
        feats.extend([np.mean(np.abs(sig)), np.std(sig)])
    feats.extend([np.corrcoef(imu_data[:,0],imu_data[:,1])[0,1],np.corrcoef(imu_data[:,0],imu_data[:,2])[0,1],np.corrcoef(imu_data[:,1],imu_data[:,2])[0,1]])
    return np.array(feats)
X_imu_tr = np.array([extract_imu(np.load(f'{imu_dir}/train/{s}/{s}.npy',allow_pickle=True)) for s in train_ids])
X_imu_te = np.array([extract_imu(np.load(f'{imu_dir}/{s}/{s}.npy',allow_pickle=True)) for s in test_ids])
features['IMU (36d)'] = (X_imu_tr, X_imu_te)

# 5-7. Audio/Video (load full, PCA handled in evaluation)
def load_modality(split, mod_type, subj_ids):
    is_audio = mod_type in ['opensmile','mfcc64']; sd = 'Audio' if is_audio else 'Video'
    base = f'{AB}/../{sd}/{split}/{mod_type}' if split else f'{AB}/../{sd}/{mod_type}'
    evs = ['E1','E2','E3'] if is_audio else ['event_1','event_2','event_3']
    default_dim = 65 if is_audio else 710
    feats = []
    for sid in subj_ids:
        sf = []
        for ev in evs:
            fp = f'{base}/{sid}/{ev}.npy' if is_audio else f'{base}/{sid}/{ev}/{ev}_all.npy'
            if not os.path.exists(fp) and not is_audio: fp = f'{base}/{sid}/{ev}.npy'
            if os.path.exists(fp):
                f = np.load(fp, allow_pickle=True); sf.append(f.mean(axis=0) if f.ndim==2 else f)
        feats.append(np.mean(sf,axis=0) if len(sf)==3 else np.zeros(default_dim))
    return np.array(feats)

features['OpenSmile (65d)'] = (load_modality('train','opensmile',train_ids), load_modality(None,'opensmile',test_ids))
features['MFCC64 (64d)'] = (load_modality('train','mfcc64',train_ids), load_modality(None,'mfcc64',test_ids))
features['OpenFace (710d)'] = (load_modality('train','openface',train_ids), load_modality(None,'openface',test_ids))

# 8. emo2vec mean
emo_tr = np.array([np.load(f'{W}/features/emotion2vec_train/layer_mean/{s}.npy') for s in train_ids])
emo_te = np.array([np.load(f'{W}/features/emotion2vec_test/layer_mean/{s}.npy') for s in test_ids])
features['emo2vec mean (768d)'] = (emo_tr, emo_te)

# 9-12. emo2vec different layers
for layer in ['prenet', '0', '4', '7', 'final']:
    l_tr = np.array([np.load(f'{W}/features/emotion2vec_train/layer_{layer}/{s}.npy') for s in train_ids])
    l_te = np.array([np.load(f'{W}/features/emotion2vec_test/layer_{layer}/{s}.npy') for s in test_ids])
    features[f'emo2vec {layer} (768d)'] = (l_tr, l_te)

# 13-14. emo2vec per-segment
for seg in ['seg1', 'seg2', 'seg3']:
    seg_dir = f'{W}/features/emotion2vec_segments/layer_mean'
    s_tr = []
    for s in train_ids:
        fp = f'{seg_dir}/{s}/event_{seg[-1]}.npy' if os.path.exists(f'{seg_dir}/{s}/event_{seg[-1]}.npy') else None
        s_tr.append(np.load(fp) if fp else np.zeros(768))
    s_te = []
    for s in test_ids:
        fp = f'{seg_dir}/{s}/event_{seg[-1]}.npy' if os.path.exists(f'{seg_dir}/{s}/event_{seg[-1]}.npy') else None
        s_te.append(np.load(fp) if fp else np.zeros(768))
    features[f'emo2vec {seg} (768d)'] = (np.array(s_tr), np.array(s_te))

# 15. emo2vec bootstrap-stable top-50
# Compute stability on training set
def bootstrap_stable(X, y, n_bootstrap=200):
    stable = []
    for d in range(X.shape[1]):
        signs = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(y), len(y), replace=True)
            if np.std(X[idx, d]) < 1e-8: continue
            corr = np.corrcoef(X[idx, d], y[idx])[0, 1]
            signs.append(np.sign(corr))
        if len(signs) > 0 and (len(set(signs)) == 1 or max(signs.count(s) for s in set(signs))/len(signs) >= 0.9):
            stable.append(d)
    return stable[:min(50, len(stable))]

emo_scl = StandardScaler()
emo_tr_s = emo_scl.fit_transform(emo_tr)
stable_idx = bootstrap_stable(emo_tr_s, y2)
if stable_idx:
    features[f'emo2vec stable({len(stable_idx)}d)'] = (emo_tr[:, stable_idx], emo_te[:, stable_idx])

# 16. OpenFace AU only (last ~35 columns: AU intensity + presence)
au_cols = list(range(675, 710))  # approximate AU positions
features['OpenFace AU (35d)'] = (
    load_modality('train','openface',train_ids)[:, au_cols],
    load_modality(None,'openface',test_ids)[:, au_cols])

# 17. ASR BERT features (if available)
asr_feat_file = f'{W}/features/text_embeddings/train_mean_embeddings.npy'
if os.path.exists(asr_feat_file):
    asr_tr = np.load(asr_feat_file)
    asr_te = np.load(f'{W}/features/text_embeddings/test_mean_embeddings.npy') if os.path.exists(f'{W}/features/text_embeddings/test_mean_embeddings.npy') else None
    asr_ids = np.load(f'{W}/features/text_embeddings/train_ids.npy')
    # Align with train_ids
    asr_map = {int(asr_ids[i]): asr_tr[i] for i in range(len(asr_ids))}
    asr_tr_aligned = np.array([asr_map.get(s, np.zeros(768)) for s in train_ids])
    if asr_te is not None:
        features['ASR BERT (768d)'] = (asr_tr_aligned, asr_te)

# ====== EVALUATE ======
print(f"{'Feature':<35} {'Dim':<8} {'Best Acc':<10} {'Best F1':<10} {'C':<10} {'PCA':<8}")
print("-" * 85)

results = []
for name, (Xtr, Xte) in features.items():
    if Xtr.shape[0] < 80: continue  # skip incomplete
    ba, bf, C, pca_str = best_lr_binary(Xtr, Xte, name)
    dim = Xtr.shape[1]
    results.append((name, dim, ba, bf, C, pca_str))

results.sort(key=lambda x: x[2], reverse=True)

for name, dim, ba, bf, C, pca_str in results:
    print(f"{name:<35} {dim:<8} {ba:<10.4f} {bf:<10.4f} {C:<10.4f} {pca_str:<8}")

# Save
with open(f'{W}/results/all_features_benchmark.json', 'w') as f:
    json.dump([{'name':n,'dim':d,'acc':ba,'f1':bf,'C':C,'pca':pc} for n,d,ba,bf,C,pc in results], f, indent=2)
print(f"\nSaved {len(results)} results to results/all_features_benchmark.json")
