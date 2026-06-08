"""Run ch20's classifiers (Python equivalents) on the 4 imbalance-sweep datasets,
then score/plot them with imcp's own functions.

Neural nets are REAL PyTorch here (not sklearn MLPClassifier), wrapped in a tiny
sklearn-compatible class so they slot into the same pipeline / imcp flow:
  kNN (weighted)     <- LearnerClassifKKNN (kknn)
  L1-Logistic (CV)   <- LearnerClassifCVGlmnet (glmnet::cv.glmnet)   [saga = multinomial L1]
  NN linear (torch)  <- mlr3torch torch_linear     (no hidden layer = softmax regression)
  NN dense-50 (torch)<- mlr3torch torch_dense_50    (one hidden layer of 50 ReLU units)
  Featureless        <- LearnerClassifFeatureless   (predict class priors)

Cloud host setup:
    pip install torch numpy pandas scikit-learn matplotlib
    # plus the imcp package: run this script from inside the imcp repo, or `pip install .`
Auto-uses CUDA if available. Methodology: stratified 70/30 train/test split,
predict_proba on the held-out test set, fed to imcp.
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")                  # set backend before imcp pulls in pyplot

import numpy as np
# numpy >= 2.3 removed np.trapz (renamed np.trapezoid); imcp's mcp_score/imcp_score
# still call np.trapz. Re-alias so this runs on any numpy the cloud host installs.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

import pandas as pd
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from imcp import mcp_score, imcp_score, plot_mcp_curve, plot_imcp_curve

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUT = os.path.join(REPO, "classifier_imcp_out")
os.makedirs(OUT, exist_ok=True)
TOL = 1e-6
RNG = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = {
    "A_exp-0 (100/100/100)":    "tests/A_exp-0.base-100.csv",
    "B_exp-2 (100/400/900)":    "tests/B_exp-2.base-100.csv",
    "C_exp-4 (100/1600/8100)":  "tests/C_exp-4.base-100.csv",
    "D_exp-6 (100/6400/72900)": "tests/D_exp-6.base-100.csv",
}


class TorchClassifier(BaseEstimator, ClassifierMixin):
    """Minimal real-PyTorch classifier with an sklearn fit/predict_proba interface.

    hidden=()    -> linear softmax model    (mirrors mlr3torch `torch_linear`)
    hidden=(50,) -> one 50-unit ReLU layer  (mirrors mlr3torch `torch_dense_50`)
    """

    def __init__(self, hidden=(), epochs=100, min_steps=2000, lr=1e-2, batch_size=256,
                 weight_decay=0.0, device=DEVICE, random_state=RNG):
        self.hidden = hidden
        self.epochs = epochs
        self.min_steps = min_steps   # floor on total gradient steps (helps tiny datasets)
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.random_state = random_state

    def _build(self, n_features, n_classes):
        layers, prev = [], n_features
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_classes)]   # logits; CrossEntropyLoss adds softmax
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.fromiter((idx[v] for v in y), dtype=np.int64, count=len(y))

        Xt = torch.tensor(np.asarray(X, dtype=np.float32))
        yt = torch.tensor(y_idx)
        self.model_ = self._build(Xt.shape[1], len(self.classes_)).to(self.device)

        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        gen = torch.Generator().manual_seed(self.random_state)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=self.batch_size, shuffle=True, generator=gen,
        )

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss_fn(self.model_(xb), yb).backward()
                opt.step()
        return self

    def predict_proba(self, X):
        self.model_.eval()
        Xt = torch.tensor(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            proba = torch.softmax(self.model_(Xt), dim=1).double().cpu().numpy()
        proba /= proba.sum(axis=1, keepdims=True)   # guarantee rows sum to 1 for imcp
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def make_classifiers():
    return {
        "kNN (weighted)": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(weights="distance", n_jobs=-1),
        ),
        "L1-Logistic (CV)": make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(penalty="l1", solver="saga", Cs=5, cv=3,
                                 max_iter=1000, tol=1e-3, n_jobs=-1, random_state=RNG),
        ),
        "NN linear (torch)": make_pipeline(
            StandardScaler(),
            TorchClassifier(hidden=(), epochs=100),
        ),
        "NN dense-50 (torch)": make_pipeline(
            StandardScaler(),
            TorchClassifier(hidden=(50,), epochs=100, weight_decay=1e-4),
        ),
        "Featureless": DummyClassifier(strategy="prior"),
    }


def main():
    print(f"torch {torch.__version__} | device = {DEVICE} | numpy {np.__version__}")
    rows = []
    for dsname, relpath in DATASETS.items():
        df = pd.read_csv(os.path.join(REPO, relpath), sep="\t")
        X = df[["data_1", "data_2"]].to_numpy()
        y = LabelEncoder().fit_transform(df["classes"].to_numpy())  # 'A','B','C' -> 0,1,2
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=RNG
        )

        scores = {}
        for cname, clf in make_classifiers().items():
            clf.fit(Xtr, ytr)
            proba = clf.predict_proba(Xte)        # columns ordered by sorted classes
            scores[cname] = proba
            rows.append({
                "dataset": dsname,
                "classifier": cname,
                "AU(MCP)":  round(float(mcp_score(yte, proba, abs_tolerance=TOL)), 4),
                "AU(IMCP)": round(float(imcp_score(yte, proba, abs_tolerance=TOL)), 4),
            })

        tag = dsname.split()[0]
        plot_mcp_curve(yte, scores, abs_tolerance=TOL,
                       output_fig_path=os.path.join(OUT, f"{tag}_mcp.png"))
        plot_imcp_curve(yte, scores, abs_tolerance=TOL,
                        output_fig_path=os.path.join(OUT, f"{tag}_imcp.png"))
        print(f"done: {dsname}")

    res = pd.DataFrame(rows)
    res["gap (MCP-IMCP)"] = (res["AU(MCP)"] - res["AU(IMCP)"]).round(4)
    res.to_csv(os.path.join(OUT, "results.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n================ AU(MCP) vs AU(IMCP) ================")
    print(res.to_string(index=False))
    print(f"\nSaved 10 figures + results.csv to {OUT}")


if __name__ == "__main__":
    main()
