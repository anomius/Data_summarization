import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance, gaussian_kde
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_rmse(real: np.ndarray, synth: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(real, synth))


def compute_emd(real: np.ndarray, synth: np.ndarray) -> float:
    return float(wasserstein_distance(real, synth))


def compute_nndr(real_df: pd.DataFrame, synth_df: pd.DataFrame, cols: List[str]) -> float:
    real_X = real_df[cols].dropna().to_numpy()
    synth_X = synth_df[cols].dropna().to_numpy()
    if real_X.shape[0] < 1 or synth_X.shape[0] < 2:
        return float("nan")
    nn_real = NearestNeighbors(n_neighbors=1).fit(real_X)
    d_real, _ = nn_real.kneighbors(synth_X)
    nn_syn = NearestNeighbors(n_neighbors=2).fit(synth_X)
    d_syn, _ = nn_syn.kneighbors(synth_X)
    ratio = d_real[:, 0] / (d_syn[:, 1] + np.finfo(float).eps)
    return float(np.mean(ratio))


def generate_plotly_dashboard(df1: pd.DataFrame, df2: pd.DataFrame, output_path: str, file1: str, file2: str):
    common_cols = list(set(df1.select_dtypes(include=np.number).columns)
                       & set(df2.select_dtypes(include=np.number).columns))

    metrics_log: Dict[str, Dict[str, float]] = {}
    total_rows = len(common_cols) * 3

    fig = make_subplots(
        rows=total_rows, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.03,
        subplot_titles=[
            f"{col} - Histogram" if i % 3 == 0 else
            f"{col} - KDE" if i % 3 == 1 else
            f"{col} - Metrics"
            for col in common_cols for i in range(3)
        ]
    )

    for idx, col in enumerate(common_cols):
        base_row = idx * 3 + 1
        real_series = df1[col].dropna()
        synth_series = df2[col].dropna()
        if real_series.empty or synth_series.empty:
            continue

        # Histogram
        fig.add_trace(go.Histogram(x=real_series, name=f"{col} Real", opacity=0.6), row=base_row, col=1)
        fig.add_trace(go.Histogram(x=synth_series, name=f"{col} Synthetic", opacity=0.6), row=base_row, col=1)

        # KDE
        kde_x = np.linspace(min(real_series.min(), synth_series.min()),
                            max(real_series.max(), synth_series.max()), 200)
        kde_real = gaussian_kde(real_series)(kde_x)
        kde_synth = gaussian_kde(synth_series)(kde_x)
        fig.add_trace(go.Scatter(x=kde_x, y=kde_real, mode='lines', name=f"{col} Real KDE"), row=base_row + 1, col=1)
        fig.add_trace(go.Scatter(x=kde_x, y=kde_synth, mode='lines', name=f"{col} Synthetic KDE"), row=base_row + 1, col=1)

        # Metrics
        rmse = compute_rmse(real_series.to_numpy(), synth_series.to_numpy())
        emd = compute_emd(real_series.to_numpy(), synth_series.to_numpy())
        nndr = compute_nndr(df1, df2, [col])
        metrics_log[col] = {"RMSE": rmse, "EMD": emd, "NNDR": nndr}
        fig.add_trace(go.Bar(x=["RMSE", "EMD", "NNDR"],
                             y=[rmse, emd, nndr],
                             name=f"{col} Metrics",
                             marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"]),
                      row=base_row + 2, col=1)

    fig.update_layout(
        height=350 * len(common_cols),
        width=1000,
        title_text=f"Stacked Dashboard: {os.path.basename(file1)} vs {os.path.basename(file2)}",
        barmode='overlay',
        showlegend=False
    )

    html_path = os.path.join(output_path, "plotly_stacked_dashboard.html")
    fig.write_html(html_path)
    return html_path, metrics_log


def write_metrics_csv(metrics_log: Dict[str, Dict[str, float]], output_path: str, file1: str, file2: str):
    csv_path = os.path.join(output_path, "evaluation_metrics_log.csv")
    rows = []
    for col, metrics in metrics_log.items():
        row = {
            "column": col,
            "RMSE": metrics["RMSE"],
            "EMD": metrics["EMD"],
            "NNDR": metrics["NNDR"],
            "original_file": os.path.basename(file1),
            "synthetic_file": os.path.basename(file2)
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def main(file1: str, file2: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    dashboard_path, metrics = generate_plotly_dashboard(df1, df2, output_dir, file1, file2)
    metrics_csv = write_metrics_csv(metrics, output_dir, file1, file2)

    print(f"✅ Dashboard saved to: {dashboard_path}")
    print(f"✅ Metrics log saved to: {metrics_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotly dashboard comparison of real vs synthetic data.")
    parser.add_argument("--file1", required=True, help="Path to original CSV file")
    parser.add_argument("--file2", required=True, help="Path to synthetic CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save dashboard and metrics")

    args = parser.parse_args()
    main(args.file1, args.file2, args.output_dir)