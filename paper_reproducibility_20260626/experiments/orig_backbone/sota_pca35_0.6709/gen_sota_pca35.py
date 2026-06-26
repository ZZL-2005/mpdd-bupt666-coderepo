import sys; sys.path.insert(0, '/data/zilu/mpdd2026/observation/scripts')
import numpy as np, pandas as pd, os, re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mord import OrdinalRidge
from generate_submission_ordinalridge import extract_fast_imu_features

test_ids = [1,5,7,13,15,22,28,33,34,40,42,44,47,58,74,83,85,89,90,93,105,110]
AB = '/data/zilu/mpdd2026/datasets/MPDD-AVG-2026/Young'
labels = pd.read_csv(f'{AB}/split_labels_train.csv'); train_ids = sorted(labels['ID'].values)
yp = labels.set_index('ID').loc[train_ids, 'phq9_score'].values

desc = pd.read_csv(f'{AB}/descriptions.csv')
def bfi(text):
    s={}
    for d,p in [('E','Extraversion score of (\\d+)'),('A','Agreeableness score of (\\d+)'),
                ('C','Conscientiousness score of (\\d+)'),('N','Neuroticism score of (\\d+)'),
                ('O','Openness score of (\\d+)')]:
        m=re.search(p,str(text));s[d]=int(m.group(1))/10.0 if m else 0.5
    return pd.Series(s)
pers = desc.set_index('ID')['Descriptions'].apply(bfi)
Xp_tr = pers.loc[train_ids].values.astype(float); Xp_te = pers.loc[test_ids].values.astype(float)

imu_dir = f'{AB}/IMU'
Xi_tr = np.array([list(extract_fast_imu_features(np.load(f'{imu_dir}/train/{s}/{s}.npy',allow_pickle=True)).values()) for s in train_ids])
Xi_te = np.array([list(extract_fast_imu_features(np.load(f'{imu_dir}/{s}/{s}.npy',allow_pickle=True)).values()) for s in test_ids])

Xall_tr = np.hstack([Xp_tr, Xi_tr]); Xall_te = np.hstack([Xp_te, Xi_te])
sc = StandardScaler(); Xt = sc.fit_transform(Xall_tr); Xv = sc.transform(Xall_te)
pca = PCA(35); Xt = pca.fit_transform(Xt); Xv = pca.transform(Xv)

m = OrdinalRidge(alpha=1.0).fit(Xt, yp)
pp = np.clip(m.predict(Xv), 0, 27)
bp = (pp >= 5).astype(int); tp = np.where(pp < 5, 0, np.where(pp < 10, 1, 2))

out = '/data/zilu/mpdd2026/workplace/submissions/sota_pca35'
os.makedirs(out, exist_ok=True)
with open(f'{out}/binary.csv','w') as f:
    f.write('id,binary_pred,phq9_pred\n')
    for s,b,p in zip(test_ids,bp,pp): f.write(f'{s},{b},{p:.1f}\n')
with open(f'{out}/ternary.csv','w') as f:
    f.write('id,ternary_pred,phq9_pred\n')
    for s,t,p in zip(test_ids,tp,pp): f.write(f'{s},{t},{p:.1f}\n')
import subprocess; subprocess.run(['zip','-q','-j',f'{out}/submission.zip',f'{out}/binary.csv',f'{out}/ternary.csv'])
print(f"Saved: {out}/submission.zip")
print(f"PHQ9=[{pp.min():.1f},{pp.max():.1f}], Bin={np.bincount(bp)}, Ter={np.bincount(tp,minlength=3)}")
