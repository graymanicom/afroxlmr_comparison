import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

def auc_probe_ngram(texts, y):
    vec = TfidfVectorizer(ngram_range=(1,3), min_df=3, max_df=0.9)
    X = vec.fit_transform(texts)
    y = np.asarray(y, dtype=int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in cv.split(X, y):
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))