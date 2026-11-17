"""
run_experiments.py
------------------
Repeated stratified k-fold comparison of Scikit-learn, XGBoost and PyTorch
classifiers on two benchmark datasets that ship with scikit-learn (no download):

  1. Breast Cancer Wisconsin (Diagnostic)  - 569 x 30, binary (malignant = positive)
  2. Optical Recognition of Handwritten Digits (UCI, 8x8 images) - 1797 x 64, 10 classes

Usage:
    pip install numpy scipy pandas scikit-learn matplotlib xgboost torch
    python run_experiments.py --repeats 10        # recommended for the paper
    python run_experiments.py --repeats 3         # quick test

Output: results.json (everything build_paper.py needs).
Models whose library is not installed are skipped and recorded as "not run".
"""
import argparse, json, platform, time, warnings
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, matthews_corrcoef, confusion_matrix)

warnings.filterwarnings("ignore")
SEED = 42

try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None


# ----------------------------------------------------------------------------
# PyTorch MLP wrapped as a scikit-learn estimator (so all models share one protocol)
# ----------------------------------------------------------------------------
if torch is not None:
    class TorchMLP(BaseEstimator, ClassifierMixin):
        """Same architecture/hyper-parameters as sklearn MLPClassifier below
        (64-32 ReLU, Adam lr=1e-3, batch 32, 100 epochs, no dropout, no weight decay)."""
        def __init__(self, hidden=(64, 32), lr=1e-3, epochs=100, batch_size=32, random_state=SEED):
            self.hidden, self.lr, self.epochs = hidden, lr, epochs
            self.batch_size, self.random_state = batch_size, random_state

        def fit(self, X, y):
            torch.manual_seed(self.random_state)
            self.classes_ = np.unique(y)
            yi = np.searchsorted(self.classes_, y)
            layers, d = [], X.shape[1]
            for h in self.hidden:
                layers += [nn.Linear(d, h), nn.ReLU()]
                d = h
            layers.append(nn.Linear(d, len(self.classes_)))
            self.net_ = nn.Sequential(*layers)
            opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
            lossf = nn.CrossEntropyLoss()
            Xt = torch.tensor(X, dtype=torch.float32)
            yt = torch.tensor(yi, dtype=torch.long)
            g = torch.Generator().manual_seed(self.random_state)
            self.net_.train()
            for _ in range(self.epochs):
                perm = torch.randperm(len(Xt), generator=g)
                for i in range(0, len(Xt), self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    opt.zero_grad()
                    lossf(self.net_(Xt[idx]), yt[idx]).backward()
                    opt.step()
            return self

        def predict_proba(self, X):
            self.net_.eval()
            with torch.no_grad():
                return torch.softmax(self.net_(torch.tensor(X, dtype=torch.float32)), 1).numpy()

        def predict(self, X):
            return self.classes_[self.predict_proba(X).argmax(1)]


def scaled(est):
    return Pipeline([("sc", StandardScaler()), ("clf", est)])


def build_models():
    """name -> (framework, algorithm, estimator, param_grid or None)"""
    m = {
        "Logistic Regression": ("Scikit-learn", "Linear model",
                                scaled(LogisticRegression(max_iter=2000, random_state=SEED)),
                                {"clf__C": [0.1, 1, 10]}),
        "SVM (RBF)": ("Scikit-learn", "Kernel method",
                      scaled(SVC(kernel="rbf", probability=True, random_state=SEED)),
                      {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]}),
        "Random Forest": ("Scikit-learn", "Bagged trees",
                          RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
                          {"max_depth": [None, 10], "min_samples_leaf": [1, 3]}),
        "MLP (scikit-learn)": ("Scikit-learn", "Neural network",
                               scaled(MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                                    solver="adam", learning_rate_init=1e-3, alpha=0.0,
                                                    batch_size=32, max_iter=100, random_state=SEED)),
                               None),
    }
    if xgb is not None:
        m["XGBoost"] = ("XGBoost", "Boosted trees",
                        xgb.XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=SEED,
                                          n_jobs=1, verbosity=0),
                        {"max_depth": [3, 5], "learning_rate": [0.05, 0.1]})
    if torch is not None:
        m["MLP (PyTorch)"] = ("PyTorch", "Neural network", scaled(TorchMLP()), None)
    return m


def fold_metrics(y, pred, proba, classes):
    binary = len(classes) == 2
    acc = accuracy_score(y, pred)
    if binary:  # malignant (label 0) is the positive class
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0], average=None, zero_division=0)
        p, r, f = float(p[0]), float(r[0]), float(f[0])
        auc = roc_auc_score((y == 0).astype(int), proba[:, 0])
        tn, fp, fn, tp = confusion_matrix(y == 0, pred == 0).ravel()
        extra = {"fn": int(fn), "fp": int(fp)}
    else:
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
        extra = {}
    return dict(acc=acc, prec=float(p), rec=float(r), f1=float(f), mcc=matthews_corrcoef(y, pred),
                auc=float(auc), **extra)


def run_dataset(name, X, y, n_splits, repeats):
    models = build_models()
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=SEED)
    res = {m: {"framework": v[0], "algorithm": v[1], "folds": []} for m, v in models.items()}
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    n_folds = n_splits * repeats
    for k, (tr, te) in enumerate(rskf.split(X, y), 1):
        print(f"  [{name}] fold {k}/{n_folds}", flush=True)
        for mname, (_, _, est, grid) in models.items():
            est = clone(est)
            if grid:
                gs = GridSearchCV(est, grid, cv=inner, scoring="accuracy", n_jobs=1, refit=True)
                gs.fit(X[tr], y[tr])
                fit_time, model, best = gs.refit_time_, gs.best_estimator_, str(gs.best_params_)
            else:
                t0 = time.perf_counter(); est.fit(X[tr], y[tr]); fit_time = time.perf_counter() - t0
                model, best = est, "fixed"
            t0 = time.perf_counter(); pred = model.predict(X[te]); inf_time = time.perf_counter() - t0
            proba = model.predict_proba(X[te])
            d = fold_metrics(y[te], pred, proba, np.unique(y))
            d.update(train_time=fit_time, infer_time=inf_time, best=best)
            res[mname]["folds"].append(d)
    return res


def summarise(res):
    keys = ["acc", "prec", "rec", "f1", "mcc", "auc", "train_time", "infer_time", "fn"]
    for m, v in res.items():
        v["summary"] = {}
        for k in keys:
            if k in v["folds"][0]:
                a = np.array([f[k] for f in v["folds"]], dtype=float)
                v["summary"][k] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
                                   "min": float(a.min()), "max": float(a.max())}


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i]); adj[i] = min(1.0, running)
    return adj


def statistics(res, n_train, n_test):
    """Friedman test over folds + Nadeau-Bengio corrected resampled t-test vs best model (Holm-adjusted)."""
    names = list(res)
    A = np.array([[f["acc"] for f in res[m]["folds"]] for m in names])  # models x folds
    out = {"friedman_p": None, "best": None, "vs_best": {}}
    if len(names) >= 3:
        out["friedman_p"] = float(stats.friedmanchisquare(*A)[1])
    best = int(A.mean(1).argmax()); out["best"] = names[best]
    k = A.shape[1]; others = [i for i in range(len(names)) if i != best]; pv = []
    for i in others:
        d = A[best] - A[i]; var = d.var(ddof=1)
        if var == 0:
            pv.append(1.0 if d.mean() == 0 else 0.0); continue
        t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
        pv.append(float(2 * stats.t.sf(abs(t), k - 1)))
    adj = holm(np.array(pv)) if pv else []
    for i, p, pa in zip(others, pv, adj):
        out["vs_best"][names[i]] = {"mean_diff": float((A[best] - A[i]).mean()), "p": p, "p_holm": float(pa)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    datasets = {}
    bc = load_breast_cancer(); datasets["Breast Cancer Wisconsin (Diagnostic)"] = (bc.data, bc.target, "binary")
    dg = load_digits();        datasets["Handwritten Digits (8x8)"] = (dg.data, dg.target, "multiclass")

    out = {"config": {"splits": a.splits, "repeats": a.repeats, "seed": SEED},
           "env": {"python": platform.python_version(), "platform": platform.platform(),
                   "processor": platform.processor() or platform.machine()},
           "datasets": {}}
    import sklearn, scipy
    out["env"].update(sklearn=sklearn.__version__, scipy=scipy.__version__, numpy=np.__version__,
                      xgboost=getattr(xgb, "__version__", None), torch=getattr(torch, "__version__", None))
    out["not_run"] = [n for n, lib in (("XGBoost", xgb), ("MLP (PyTorch)", torch)) if lib is None]

    for dname, (X, y, kind) in datasets.items():
        print(f"Dataset: {dname}", flush=True)
        res = run_dataset(dname, X, y, a.splits, a.repeats)
        summarise(res)
        n = len(y); n_test = n / a.splits
        out["datasets"][dname] = {
            "kind": kind, "n": int(n), "features": int(X.shape[1]), "classes": int(len(np.unique(y))),
            "class_counts": {int(c): int((y == c).sum()) for c in np.unique(y)},
            "models": res, "stats": statistics(res, n - n_test, n_test)}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
