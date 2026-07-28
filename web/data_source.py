"""
Data layer for the OnionGuard web app.

Everything the web pages need is behind the DataSource interface, so the
Flask app never reads a CSV path or a Supabase table directly. Today the
only implementation is LocalFileDataSource (reads data/*.csv and
reports/*), which is all Phase 6 needs while the project is still on mock
data. When Phase 5 (Supabase) happens, add a SupabaseDataSource with the
same methods and change get_data_source() — no route or template changes.

Every method returns plain Python dicts/lists (never a DataFrame), because
that's the shape a Supabase client would return too. Keeping the return
contract database-shaped now is what makes the later swap mechanical.
"""
import csv
from abc import ABC, abstractmethod
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

# Feature columns surfaced in the /samples table. Deliberately a short list
# of the ones Phase 3 found most important — features.csv has 46, which is
# unreadable in a table. Add/remove freely; templates render whatever is here.
TABLE_FEATURE_COLUMNS = [
    ("n_small_sharp_blobs_viewmean", "จุดเล็กคมชัด (เฉลี่ย)", 2),
    ("n_small_sharp_blobs_viewmax", "จุดเล็กคมชัด (สูงสุด)", 2),
    ("cluster_density_viewmean", "ความหนาแน่นกลุ่มจุด", 4),
    ("n_large_blotches_viewmean", "ปื้นใหญ่ (เฉลี่ย)", 2),
    ("texture_viewmean", "พื้นผิว", 4),
]


def _read_csv_dicts(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class DataSource(ABC):
    """Read contract for every page. A Supabase implementation only needs
    to satisfy these five methods."""

    @abstractmethod
    def get_samples(self):
        """One dict per head: sample_code, compactdry (ground truth, may be
        None once real unlabelled scans exist), pred_label, pred_proba, and
        the TABLE_FEATURE_COLUMNS values."""

    @abstractmethod
    def get_dataset_stats(self):
        """Class counts / balance for /dataset."""

    @abstractmethod
    def get_model_metrics(self):
        """Per-fold + mean/SD metrics and confusion matrix for /model."""

    @abstractmethod
    def get_model_info(self):
        """Hyperparameters, threshold, feature count from model_config.json."""

    @abstractmethod
    def is_mock_data(self):
        """True while the dataset is synthetic — drives the warning banner
        shown on every page."""


class LocalFileDataSource(DataSource):
    """Reads the artifacts the Phase 1-4 scripts already write to disk."""

    def __init__(self, data_dir=DATA_DIR, reports_dir=REPORTS_DIR, models_dir=MODELS_DIR):
        self.data_dir = Path(data_dir)
        self.reports_dir = Path(reports_dir)
        self.models_dir = Path(models_dir)

    def get_samples(self):
        features = _read_csv_dicts(self.data_dir / "features.csv")
        oof = {r["sample_code"]: r for r in _read_csv_dicts(self.reports_dir / "phase3_oof_predictions.csv")}

        samples = []
        for row in features:
            code = row["sample_code"]
            pred = oof.get(code)
            truth_raw = row.get("compactdry", "")
            entry = {
                "sample_code": code,
                "compactdry": int(truth_raw) if truth_raw not in ("", None) else None,
                "pred_label": int(pred["y_pred"]) if pred else None,
                "pred_proba": _to_float(pred["y_proba"]) if pred else None,
                "features": {},
            }
            # Agreement flag drives the "ตรงกัน / ไม่ตรงกัน" column; None when
            # either side is missing rather than silently counting as a match.
            if entry["compactdry"] is not None and entry["pred_label"] is not None:
                entry["correct"] = entry["compactdry"] == entry["pred_label"]
            else:
                entry["correct"] = None

            for col, _label, _places in TABLE_FEATURE_COLUMNS:
                entry["features"][col] = _to_float(row.get(col))
            samples.append(entry)
        return samples

    def get_dataset_stats(self):
        samples = self.get_samples()
        total = len(samples)
        n_pos = sum(1 for s in samples if s["compactdry"] == 1)
        n_neg = sum(1 for s in samples if s["compactdry"] == 0)
        n_unlabelled = sum(1 for s in samples if s["compactdry"] is None)

        n_views = 0
        images_dir = ROOT / "images"
        if images_dir.exists():
            n_views = len(list(images_dir.glob("*.png")))

        labelled = n_pos + n_neg
        return {
            "total_samples": total,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_unlabelled": n_unlabelled,
            "pct_positive": (n_pos / labelled * 100) if labelled else 0.0,
            "pct_negative": (n_neg / labelled * 100) if labelled else 0.0,
            # Ratio of the smaller class to the larger; 1.0 = perfectly balanced.
            "balance_ratio": (min(n_pos, n_neg) / max(n_pos, n_neg)) if max(n_pos, n_neg) else 0.0,
            "n_images": n_views,
            "n_views_per_sample": (n_views // total) if total else 0,
        }

    def get_model_metrics(self):
        folds = _read_csv_dicts(self.reports_dir / "phase3_cv_fold_metrics.csv")
        metric_keys = ["accuracy", "precision", "recall", "specificity", "f1", "kappa"]

        fold_rows, summary = [], {}
        for r in folds:
            fold_rows.append({
                "fold": int(_to_float(r.get("fold"))),
                "n_train": int(_to_float(r.get("n_train"))),
                "n_test": int(_to_float(r.get("n_test"))),
                "oob_score": _to_float(r.get("oob_score")),
                **{k: _to_float(r.get(k)) for k in metric_keys},
            })

        for k in metric_keys:
            vals = [f[k] for f in fold_rows]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                summary[k] = {"mean": mean, "sd": var ** 0.5}
            else:
                summary[k] = {"mean": 0.0, "sd": 0.0}

        # Confusion matrix rebuilt from OOF predictions rather than parsed out
        # of the PNG, so the numbers on the page are real data, not a caption.
        oof = _read_csv_dicts(self.reports_dir / "phase3_oof_predictions.csv")
        tn = sum(1 for r in oof if r["y_true"] == "0" and r["y_pred"] == "0")
        fp = sum(1 for r in oof if r["y_true"] == "0" and r["y_pred"] == "1")
        fn = sum(1 for r in oof if r["y_true"] == "1" and r["y_pred"] == "0")
        tp = sum(1 for r in oof if r["y_true"] == "1" and r["y_pred"] == "1")

        importance = _read_csv_dicts(self.reports_dir / "phase3_feature_importance.csv")
        top_features = [
            {"feature": r["feature"], "importance": _to_float(r["mean_importance"])}
            for r in importance[:12]
        ]

        return {
            "folds": fold_rows,
            "summary": summary,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "top_features": top_features,
            "n_folds": len(fold_rows),
        }

    def get_model_info(self):
        import json
        path = self.models_dir / "model_config.json"
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return {
            "decision_threshold": cfg.get("decision_threshold"),
            "n_features": len(cfg.get("feature_names", [])),
            "best_params": cfg.get("best_params", {}),
            "random_seed": cfg.get("random_seed"),
            "oob_score": cfg.get("oob_score"),
            "trained_on_n_samples": cfg.get("trained_on_n_samples"),
        }

    def is_mock_data(self):
        """Mock as long as the generator's debug metadata file is present —
        generate_mock_data.py writes it, and real captures never would."""
        return (self.data_dir / "mock_metadata.csv").exists()


class SupabaseDataSource(LocalFileDataSource):
    """Scan records come from Supabase; model metrics stay on disk.

    Deliberately extends the local source rather than replacing it. The scan
    table is the thing that must be shared and durable, but model metrics and
    the ROC/confusion figures are products of a training run that lives with
    the code — copying them into a database would only add a way for the two
    to disagree about which model is deployed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from supabase_client import SupabaseClient
        self.client = SupabaseClient()

    def get_samples(self):
        rows = self.client.select(
            "scans",
            columns="sample_code,captured_at,pred_label,pred_conf,pred_proba,"
                    "features,compactdry_truth,ood_status,borderline",
            order="captured_at.desc",
        )
        samples = []
        for r in rows:
            truth = r.get("compactdry_truth")
            pred = r.get("pred_label")
            entry = {
                "sample_code": r["sample_code"],
                "compactdry": int(truth) if truth is not None else None,
                "pred_label": int(pred) if pred is not None else None,
                "pred_proba": r.get("pred_proba"),
                "features": {},
                # None (not False) when either side is missing — an unlabelled
                # scan is not a wrong prediction.
                "correct": (int(truth) == int(pred)) if truth is not None and pred is not None else None,
            }
            feats = r.get("features") or {}
            for col, _label, _places in TABLE_FEATURE_COLUMNS:
                entry["features"][col] = _to_float(feats.get(col))
            samples.append(entry)
        return samples

    def is_mock_data(self):
        """Mock while the deployed model was trained on generated images.
        Real scans arriving in Supabase do not change what the model learned
        from, so this still keys off the training artefacts on disk."""
        return super().is_mock_data()


def get_data_source():
    """Single place the app resolves its backend.

    Uses Supabase when credentials are present and reachable, otherwise falls
    back to local CSVs so development works with no network and a missing
    database never takes the whole app down.
    """
    import os
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from supabase_client import load_env
        load_env()
    except Exception:  # noqa: BLE001
        return LocalFileDataSource()

    if not os.environ.get("SUPABASE_URL"):
        return LocalFileDataSource()
    try:
        return SupabaseDataSource()
    except Exception as exc:  # noqa: BLE001
        print(f"[data_source] ใช้ Supabase ไม่ได้ ({exc}) — กลับไปใช้ไฟล์ในเครื่อง")
        return LocalFileDataSource()
