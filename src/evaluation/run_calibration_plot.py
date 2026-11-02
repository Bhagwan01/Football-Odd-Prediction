import os
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ["MPLCONFIGDIR"] = str(project_root / ".matplotlib_cache")

from src.data.loader import load_processed
from src.models.trainer import prepare_data_splits, FEATURE_COLUMNS
from src.evaluation.metrics import compute_calibration_curve

saved = joblib.load(project_root / "models" / "best_model.joblib")
best_model = saved["model"]

features_df = load_processed("features_match_level")
train_df, test_df, _ = prepare_data_splits(features_df, split_year=2024)

X_test = test_df[FEATURE_COLUMNS].values
y_test = test_df["target"].values
y_prob = best_model.predict_proba(X_test)

prob_pred_home = y_prob[:, 0]
prob_bmk_home = test_df["mkt_prob_home"].values
y_home_binary = (y_test == 0).astype(int)

p_model, o_model = compute_calibration_curve(y_home_binary, prob_pred_home, n_bins=10)
p_bmk, o_bmk = compute_calibration_curve(y_home_binary, prob_bmk_home, n_bins=10)

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Diagonal)")
plt.plot(p_bmk, o_bmk, "s-", color="#e74c3c", label="Bookmaker Consensus", linewidth=2)
plt.plot(p_model, o_model, "o-", color="#2980b9", label=f"{saved['model_name']}", linewidth=2)
plt.title("Reliability Diagram (Home Win Forecast Calibration)", fontsize=14, fontweight="bold")
plt.xlabel("Mean Predicted Probability", fontsize=12)
plt.ylabel("Observed Event Frequency", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig(project_root / "reports" / "figures" / "calibration_curves.png", dpi=150, bbox_inches="tight")
print("Saved: reports/figures/calibration_curves.png")
