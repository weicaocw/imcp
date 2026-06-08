import os
import sys

import matplotlib
import pandas as pd

from imcp import mcp_score, imcp_score, plot_mcp_curve, plot_imcp_curve

matplotlib.use("Agg")

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

OUT = os.path.join(REPO, "imcp_demo_out")
os.makedirs(OUT, exist_ok=True)

TOL = 1e-6

def run(name, y_true, y_score):
    mcp = mcp_score(y_true, y_score, abs_tolerance=TOL)
    imcp = imcp_score(y_true, y_score, abs_tolerance=TOL)
    print(f"AU(MCP)  = {mcp:.4f}")
    print(f"AU(IMCP) = {imcp:.4f}")

    plot_mcp_curve(y_true, y_score, abs_tolerance=TOL,
                   output_fig_path=os.path.join(OUT, f"{name}_mcp.png"))
    plot_imcp_curve(y_true, y_score, abs_tolerance=TOL,
                    output_fig_path=os.path.join(OUT, f"{name}_imcp.png"))


# Dataset 1: 7 classes
df1 = pd.read_csv(os.path.join(REPO, "tests", "test_results.csv"))
run("test_results", df1["y_true"], df1.loc[:, "y_score_1":])

# Dataset 2: 3 classes
df2 = pd.read_csv(os.path.join(REPO, "tests", "test_imbalanced_class_probs.csv"))
run("test_imbalanced", df2["y_true"], df2.drop("y_true", axis=1))

print(f"\nSaved 4 figures to {OUT}")
