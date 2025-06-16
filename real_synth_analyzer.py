"""
real_synth_analyzer.py · v1.5  (DB-first metrics, parity & KDE)
==============================================================

Stand-alone engine for comparing **real vs synthetic** tabular data.

New in v1.5
-----------
* `_matrix()` now available class-wide – used by NNDR & DB-level metrics.
* `overall_metrics()` operates in full multivariate numeric space:
    • RMSE   – average categorical frequency-vector RMSE  
    • EMD    – Wasserstein distance between *vector-norm* distributions  
    • NNDR   – nearest-neighbour distance ratio across **all** numeric dims
* `make_db_parity()` – one dot per numeric column (mean–mean scatter, R²).  
* `make_db_kde()`    – KDE of L² norms (overall spread real vs synth).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import chisquare, gaussian_kde, wasserstein_distance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

# ───────────────────────── configuration ──────────────────────────────
@dataclass(slots=True)
class AnalyzerConfig:
    sigma_th: float = 3.0        # |z| ≥ σ → numeric outlier
    contamination: float = 0.05  # IsolationForest contamination share
    random_state: int = 0        # RNG seed
    pk_uniqueness: float = 0.98  # ≥ fraction unique → PK candidate


# ───────────────────────── helper utilities ───────────────────────────
def _is_temporal(s: pd.Series) -> bool:
    """True if *s* is datetime64 or (sampled) ISO-8601 strings."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not pd.api.types.is_object_dtype(s):
        return False
    sample = s.dropna().astype(str).head(10)
    if sample.empty:
        return False
    try:
        pd.to_datetime(sample, format="ISO8601", errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def _series_to_numeric(s: pd.Series) -> np.ndarray:
    """Temporal → int64 nanoseconds; else numeric, drop NaNs."""
    if _is_temporal(s):
        return (
            pd.to_datetime(s, errors="coerce").astype("int64").dropna().to_numpy()
        )
    return pd.to_numeric(s, errors="coerce").dropna().to_numpy()


# ─────────────────────────── main analyzer ────────────────────────────
class RealSyntheticAnalyzer:
    """Analyse *real* vs *synthetic* datasets (numerical / temporal / categorical)."""

    # ─── construction ────────────────────────────────────────────────
    def __init__(
        self,
        real: Union[str, Path, pd.DataFrame],
        synth: Union[str, Path, pd.DataFrame, None] = None,
        *,
        config: AnalyzerConfig | None = None,
    ) -> None:
        self.config = config or AnalyzerConfig()
        self._load(real, synth)

    # ─── role helpers ────────────────────────────────────────────────
    def column_role(self, col: str) -> str:
        if _is_temporal(self.real[col]):
            return "temporal"
        return "numerical" if pd.api.types.is_numeric_dtype(self.real[col]) else "categorical"

    def list_columns(self, role: str) -> List[str]:
        role = role.lower()
        if role == "temporal":
            return [c for c in self.real.columns if _is_temporal(self.real[c])]
        if role == "numerical":
            return list(self.real.select_dtypes(np.number).columns)
        if role == "categorical":
            return [
                c for c in self.real.columns
                if c not in self.list_columns("numerical") + self.list_columns("temporal")
            ]
        return []

    # ─── PK / FK discovery ───────────────────────────────────────────
    def identify_keys(self) -> pd.DataFrame:
        n = len(self.real)
        pk = [
            c for c in self.real.columns
            if self.real[c].is_unique and self.real[c].isnull().sum() == 0
        ]
        pk += [
            c for c in self.real.columns if c not in pk
            and self.real[c].nunique(dropna=False) / n >= self.config.pk_uniqueness
            and self.real[c].isnull().sum() == 0
        ]
        pk = list(dict.fromkeys(pk))
        rows = []
        for col in self.real.columns:
            is_pk = col in pk
            fk_to = None
            if not is_pk:
                col_vals = set(self.real[col].dropna().unique())
                for p in pk:
                    if col_vals.issubset(set(self.real[p].unique())):
                        fk_to = p
                        break
            rows.append({"column": col, "role": self.column_role(col), "is_pk": is_pk, "fk_to": fk_to})
        return pd.DataFrame(rows)

    # ─── categorical RMSE helper ─────────────────────────────────────
    def _categorical_rmse(self, col: str) -> float:
        vr = self.real[col].value_counts(normalize=True)
        vs = self.synth[col].value_counts(normalize=True)
        cats = sorted(set(vr.index) | set(vs.index))
        r = vr.reindex(cats, fill_value=0).to_numpy()
        s = vs.reindex(cats, fill_value=0).to_numpy()
        return float(np.sqrt(mean_squared_error(r, s)))

    # ─── numeric matrix helper ───────────────────────────────────────
    def _matrix(self, cols: list[str], df: pd.DataFrame) -> np.ndarray:
        out = []
        for c in cols:
            if _is_temporal(df[c]):
                out.append(pd.to_datetime(df[c], errors="coerce").astype("int64"))
            else:
                out.append(pd.to_numeric(df[c], errors="coerce"))
        return pd.concat(out, axis=1).dropna().to_numpy()

    # ─── DB-level metrics  (full numeric space) ───────────────────────
    def overall_metrics(self) -> pd.Series:
        num_cols = self.list_columns("numerical")
        cat_cols = self.list_columns("categorical")

        # categorical RMSE -------------------------------------------
        rmse_cat = np.nan
        if cat_cols:
            rmse_cat = np.nanmean([self._categorical_rmse(c) for c in cat_cols])

        # multivariate numeric EMD + NNDR ----------------------------
        if num_cols:
            real_mat  = self._matrix(num_cols, self.real)
            synth_mat = self._matrix(num_cols, self.synth)

            # NNDR (multi-dim)
            nndr_db = self._nndr(num_cols)

            # EMD of vector norms
            r_norm = np.linalg.norm(real_mat,  axis=1)
            s_norm = np.linalg.norm(synth_mat, axis=1)
            emd_db = float(wasserstein_distance(r_norm, s_norm))
        else:
            nndr_db = emd_db = float("nan")

        return pd.Series({
            "RMSE":    rmse_cat,   # categorical only
            "EMD":     emd_db,     # multivariate
            "NNDR":    nndr_db,    # multivariate
            "Num num": len(num_cols),
            "Num cat": len(cat_cols),
        })

    # ─── per-column metrics APIs ─────────────────────────────────────
    def metrics(self, col: str) -> Tuple[float, float, float]:
        return self._categorical_metrics(col) if self.column_role(col) == "categorical" else self._numeric_metrics(col)

    def metric_names(self, col: str) -> Tuple[str, str, str]:
        return ("TVD", "Chi²-p", "Coverage") if self.column_role(col) == "categorical" else ("RMSE", "EMD", "NNDR")

    def evaluation_log(self) -> pd.DataFrame:
        recs = []
        for c in self.real.columns.intersection(self.synth.columns):
            v1, v2, v3 = self.metrics(c)
            n1, n2, n3 = self.metric_names(c)
            recs.append({"column": c, "role": self.column_role(c), n1: v1, n2: v2, n3: v3})
        metrics_cols = [k for k in recs[0] if k not in ("column", "role")]
        return pd.DataFrame(recs)[["column", "role", *metrics_cols]].sort_values(["role", "column"], ignore_index=True)

    # ─── plot mode suggestions ───────────────────────────────────────
    def suggest_plot_modes(self, col: str) -> List[str]:
        r = self.column_role(col)
        if r == "numerical":   return ["Histogram", "KDE", "Box", "Scatter", "Line", "Area"]
        if r == "categorical": return ["Bar", "Pie"]
        if r == "temporal":    return ["TimeSeries", "Histogram"]
        return []

    # ─── per-column figure builder ───────────────────────────────────
    def make_figure(
        self,
        col: str,
        *,
        mode: str | None = None,
        highlight: bool = False,
        highlight_outliers: bool | None = None,  # legacy alias
    ) -> go.Figure:
        if highlight_outliers is not None:
            highlight = highlight_outliers

        if col not in self.real or col not in self.synth:
            raise KeyError(col)

        role = self.column_role(col)
        mode = mode or self.suggest_plot_modes(col)[0]
        if mode not in self.suggest_plot_modes(col):
            raise ValueError("Invalid plot mode")

        fig = go.Figure()

        # numeric -----------------------------------------------------
        if role == "numerical":
            if mode == "Histogram":
                fig.add_histogram(x=self.real[col], name="Real", opacity=0.6)
                fig.add_histogram(x=self.synth[col], name="Synthetic", opacity=0.6)
            elif mode == "KDE":
                xs = np.linspace(min(self.real[col].min(), self.synth[col].min()),
                                 max(self.real[col].max(), self.synth[col].max()), 200)
                fig.add_scatter(x=xs, y=gaussian_kde(self.real[col].dropna())(xs), name="Real KDE")
                fig.add_scatter(x=xs, y=gaussian_kde(self.synth[col].dropna())(xs), name="Synth KDE")
            elif mode == "Box":
                fig.add_box(y=self.real[col], name="Real"); fig.add_box(y=self.synth[col], name="Synthetic")
            elif mode in ("Scatter", "Line", "Area"):
                plot_mode = "markers" if mode == "Scatter" else "lines"
                fill_kw   = {"fill": "tozeroy"} if mode == "Area" else {}
                fig.add_scatter(x=self.real.index,  y=self.real[col],  mode=plot_mode, name="Real",  **fill_kw)
                fig.add_scatter(x=self.synth.index, y=self.synth[col], mode=plot_mode, name="Synthetic", **fill_kw)
                if highlight:
                    m_r = self._outlier_mask(self.real[col]); m_s = self._outlier_mask(self.synth[col])
                    fig.add_scatter(x=self.real.index[m_r],  y=self.real[col][m_r],  mode="markers", marker=dict(color="red", symbol="x"),     name="Real outlier")
                    fig.add_scatter(x=self.synth.index[m_s], y=self.synth[col][m_s], mode="markers", marker=dict(color="red", symbol="cross"), name="Synth outlier")

        # temporal ----------------------------------------------------
        elif role == "temporal":
            if mode == "TimeSeries":
                fig.add_scatter(x=self.real[col],  y=np.arange(len(self.real)),  mode="lines", name="Real")
                fig.add_scatter(x=self.synth[col], y=np.arange(len(self.synth)), mode="lines", name="Synthetic")
            elif mode == "Histogram":
                fig.add_histogram(x=self.real[col], name="Real", opacity=0.6)
                fig.add_histogram(x=self.synth[col], name="Synthetic", opacity=0.6)

        # categorical -------------------------------------------------
        else:
            if mode == "Bar":
                fig.add_bar(x=self.real[col].value_counts().index, y=self.real[col].value_counts(), name="Real")
                fig.add_bar(x=self.synth[col].value_counts().index, y=self.synth[col].value_counts(), name="Synthetic")
            elif mode == "Pie":
                fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]])
                fig.add_pie(labels=self.real[col].value_counts().index, values=self.real[col].value_counts(), name="Real", row=1, col=1)
                fig.add_pie(labels=self.synth[col].value_counts().index, values=self.synth[col].value_counts(), name="Synthetic", row=1, col=2)

        fig.update_layout(title=f"{col} • {mode}")
        return fig

    # ─── DB-level figures ────────────────────────────────────────────
    def make_db_parity(self) -> go.Figure:
        num_cols = self.list_columns("numerical")
        if not num_cols:
            return go.Figure().update_layout(title="No numeric columns")

        means_r = self.real[num_cols].mean()
        means_s = self.synth[num_cols].mean()
        r2      = np.corrcoef(means_r, means_s)[0, 1] ** 2

        fig = go.Figure()
        maxv = max(means_r.max(), means_s.max())
        fig.add_scatter(x=[0, maxv], y=[0, maxv], mode="lines",
                        line=dict(color="gray", dash="dash"), showlegend=False)
        fig.add_scatter(x=means_r, y=means_s, mode="markers+text",
                        text=num_cols, textposition="top center",
                        hovertemplate="%{text}<br>Real=%{x:.4f}<br>Synth=%{y:.4f}")
        fig.update_layout(xaxis_title="Column mean – Real",
                          yaxis_title="Column mean – Synthetic",
                          title=f"DB-level Parity scatter (R² ≈ {r2:.3f})",
                          height=500)
        return fig

    def make_db_kde(self) -> go.Figure:
        num_cols = self.list_columns("numerical")
        if not num_cols:
            return go.Figure().update_layout(title="No numeric columns")

        r_norm = np.linalg.norm(self._matrix(num_cols, self.real),  axis=1)
        s_norm = np.linalg.norm(self._matrix(num_cols, self.synth), axis=1)
        xs = np.linspace(min(r_norm.min(), s_norm.min()), max(r_norm.max(), s_norm.max()), 300)

        fig = go.Figure()
        fig.add_scatter(x=xs, y=gaussian_kde(r_norm)(xs), name="Real")
        fig.add_scatter(x=xs, y=gaussian_kde(s_norm)(xs), name="Synthetic")
        fig.update_layout(title="DB-level L2-norm KDE",
                          xaxis_title="Vector norm", yaxis_title="Density",
                          height=500)
        return fig

    # ─── correlation & summary ───────────────────────────────────────
    def correlation_figs(self) -> Tuple[go.Figure, go.Figure]:
        rc = self.real.select_dtypes(np.number).corr()
        sc = self.synth.select_dtypes(np.number).corr()
        fig_r = go.Figure(go.Heatmap(z=rc, x=rc.columns, y=rc.index, colorscale="Blues")).update_layout(title="Real correlation")
        fig_s = go.Figure(go.Heatmap(z=sc, x=sc.columns, y=sc.index, colorscale="Reds")).update_layout(title="Synthetic correlation")
        return fig_r, fig_s

    def summary(self) -> pd.DataFrame:
        return pd.concat([self.real.describe(include="all").T, self.synth.describe(include="all").T], axis=1, keys=["Real", "Synthetic"]).sort_index()

    # ─── anomalies & export ───────────────────────────────────────────
    def detect_anomalies(self, *, max_rows: int | None = None) -> pd.DataFrame:
        X = self.real.select_dtypes(np.number)
        if X.empty:
            return pd.DataFrame()
        iso = IsolationForest(contamination=self.config.contamination, random_state=self.config.random_state).fit(X)
        idx = np.where(iso.predict(X) == -1)[0]
        df = self.real.iloc[idx]
        return df if max_rows is None else df.head(max_rows)

    def to_png(self, fig: go.Figure, *, scale: int = 2) -> bytes:
        buf = io.BytesIO(); fig.write_image(buf, format="png", scale=scale); return buf.getvalue()

    # ─── internal metrics ─────────────────────────────────────────────
    def _numeric_metrics(self, col: str) -> Tuple[float, float, float]:
        r = _series_to_numeric(self.real[col])
        s = _series_to_numeric(self.synth[col])
        if r.size == 0 or s.size == 0:
            return float("nan"), float("nan"), float("nan")
        n = min(len(r), len(s))
        rmse = float(np.sqrt(mean_squared_error(r[:n], s[:n])))
        emd  = float(wasserstein_distance(r[:n], s[:n]))
        nndr = self._nndr([col])
        return rmse, emd, nndr

    def _categorical_metrics(self, col: str) -> Tuple[float, float, float]:
        cnt_r, cnt_s = self.real[col].value_counts(), self.synth[col].value_counts()
        cats = sorted(set(cnt_r.index) | set(cnt_s.index))
        r = cnt_r.reindex(cats, fill_value=0); s = cnt_s.reindex(cats, fill_value=0)
        p, q = r / max(r.sum(), 1), s / max(s.sum(), 1)
        tvd  = 0.5 * np.abs(p - q).sum()
        expected = s * (r.sum() / max(s.sum(), 1))
        _, chi2_p = chisquare(f_obs=r, f_exp=expected)
        coverage = len(set(cnt_r.index) & set(cnt_s.index)) / max(len(cnt_r), 1)
        return float(tvd), float(chi2_p), float(coverage)

    def _nndr(self, cols: List[str]) -> float:
        def _mat(df):
            sub = []
            for c in cols:
                sub.append(pd.to_datetime(df[c], errors="coerce").astype("int64") if _is_temporal(df[c])
                           else pd.to_numeric(df[c], errors="coerce"))
            return pd.concat(sub, axis=1).dropna().to_numpy()

        rX, sX = _mat(self.real), _mat(self.synth)
        if rX.size == 0 or len(sX) < 2:
            return float("nan")
        d_real  = NearestNeighbors(n_neighbors=1).fit(rX).kneighbors(sX, return_distance=True)[0][:, 0]
        d_synth = NearestNeighbors(n_neighbors=2).fit(sX).kneighbors(sX, return_distance=True)[0][:, 1]
        return float(np.mean(d_real / (d_synth + np.finfo(float).eps)))

    def _outlier_mask(self, s: pd.Series) -> pd.Series:
        nums = _series_to_numeric(s)
        if nums.size == 0 or np.std(nums) == 0:
            return pd.Series(False, index=s.index)
        z = (nums - nums.mean()) / nums.std()
        mask = np.abs(z) >= self.config.sigma_th
        out  = pd.Series(False, index=s.index)
        out[s.dropna().index[: len(mask)]] = mask
        return out

    # ─── loader ───────────────────────────────────────────────────────
    def _load(self, real, synth):
        self.real = pd.read_csv(real) if isinstance(real, (str, Path)) else real.copy(deep=True)
        self.synth = self.real.copy(deep=True) if synth is None else (
            pd.read_csv(synth) if isinstance(synth, (str, Path)) else synth.copy(deep=True)
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RealSyntheticAnalyzer rows={len(self.real)} cols={len(self.real.columns)}>"