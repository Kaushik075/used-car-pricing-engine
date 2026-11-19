"""Retraining pipeline with a promotion gate (champion vs challenger).

Run from the project root, with a new CSV that has the same columns as the training data:
    python -m src.retrain --data path\\to\\new_listings.csv
Add --tune for a full model search (about 5 minutes) instead of refitting the
champion's settings (about 2 minutes), and --dry-run to decide without replacing anything.

What happens:
1. The new CSV is cleaned and split 80/20 exactly like the original training data.
2. A CHALLENGER is trained on the new 80% and saved in models/candidates/<time>/ (logged to MLflow).
3. Evaluation set = the new 20%, MINUS any row the CHAMPION (the current live model) was
   trained on (we check the saved row fingerprints). So neither model has seen any
   evaluation row, and the comparison is fair.
4. Both models are scored on that same evaluation set.
5. The gate promotes the challenger only if BOTH hold:
     - its average % error (MAPE) is at least GATE_MIN_IMPROVEMENT_PP points lower, and
     - its price ranges still cover 75-85% of real prices (target 80%).
   Otherwise the champion stays.
6. The decision and the reasons go to models/promotion_log.jsonl and to the challenger's MLflow run.
   On promotion, the old champion is copied to models/archive/<time>_champion/ first.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from src.cleaning import clean, load_raw, row_hashes
from src.config import (
    GATE_COVERAGE_RANGE,
    GATE_MIN_EVAL_ROWS,
    GATE_MIN_IMPROVEMENT_PP,
    HASHES_FILE,
    METADATA_FILE,
    MODEL_DIR,
    MODEL_FILE,
    PROFILE_FILE,
)
from src.evaluate import regression_metrics
from src.train import run_training, split_data

ARTIFACTS = (MODEL_FILE, METADATA_FILE, PROFILE_FILE, HASHES_FILE)


def champion_recipe(meta: dict) -> dict:
    """The champion's tuned settings, reused to refit on new data when not doing a full search."""
    cv = {row["model"]: json.loads(row["best_params"]) for row in meta["cv_comparison"]}
    return {"lightgbm": cv["lightgbm"], "lasso_alpha": cv.get("lasso", {}).get("alpha", 1e-4)}


def evaluation_set(data_path, champion_dir: Path):
    """New data's test split, minus rows the champion was trained on. Returns (X, y, n_removed)."""
    _, X_test, _, y_test = split_data(clean(load_raw(data_path)))
    hashes_path = champion_dir / HASHES_FILE
    if not hashes_path.exists():
        raise SystemExit(f"{hashes_path} not found: the champion has no training fingerprints. "
                         "Retrain it once with `python -m src.train` (Version 3) first.")
    seen = np.isin(row_hashes(X_test.assign(price=y_test)), np.load(hashes_path))
    return X_test[~seen], y_test[~seen], int(seen.sum())


def score(model, X, y) -> dict:
    est, low, high = model.predict_range(X)
    y = np.asarray(y, float)
    m = regression_metrics(y, est)
    m["coverage"] = float(((y >= low) & (y <= high)).mean())
    return m


def decide(champion: dict, challenger: dict, n_eval: int,
           min_improvement=GATE_MIN_IMPROVEMENT_PP, coverage_range=GATE_COVERAGE_RANGE,
           min_rows=GATE_MIN_EVAL_ROWS) -> tuple[bool, list[str]]:
    """The promotion gate. Returns (promote?, reasons). Pure function: easy to test."""
    if n_eval < min_rows:
        return False, [f"only {n_eval} evaluation rows that neither model was trained on "
                       f"(need {min_rows}), so a fair comparison is not possible: champion kept"]
    gain = champion["mape_pct"] - challenger["mape_pct"]
    better = gain >= min_improvement
    lo, hi = coverage_range
    covered = lo <= challenger["coverage"] <= hi
    reasons = [
        f"accuracy: challenger MAPE {challenger['mape_pct']:.3f}% vs champion {champion['mape_pct']:.3f}% "
        f"({gain:+.3f} points; need at least +{min_improvement:.2f}) -> {'PASS' if better else 'FAIL'}",
        f"ranges: challenger covers {challenger['coverage']:.1%} of real prices "
        f"(need {lo:.0%}-{hi:.0%}) -> {'PASS' if covered else 'FAIL'}",
    ]
    return better and covered, reasons


def promote(candidate_dir: Path, model_dir: Path, stamp: str) -> Path:
    """Archive the current champion, then copy the challenger's files into the model folder."""
    archive = model_dir / "archive" / f"{stamp}_champion"
    archive.mkdir(parents=True, exist_ok=True)
    for f in ARTIFACTS:
        if (model_dir / f).exists():
            shutil.copy2(model_dir / f, archive / f)
    for f in ARTIFACTS:
        shutil.copy2(candidate_dir / f, model_dir / f)
    return archive


def retrain(data_path, model_dir=MODEL_DIR, tune=False, dry_run=False, track=True) -> dict:
    model_dir = Path(model_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    champion_meta = json.loads((model_dir / METADATA_FILE).read_text())
    candidate_dir = model_dir / "candidates" / stamp

    print(f"== Training challenger on {Path(data_path).name} ({'full search' if tune else 'champion settings'})")
    challenger_meta = run_training(
        data_path, candidate_dir, tune=tune,
        fixed_params=None if tune else champion_recipe(champion_meta),
        track=track, run_name="retrain-challenger", tags={"role": "challenger"},
    )

    print("\n== Comparing champion and challenger on the same unseen rows")
    X_eval, y_eval, n_removed = evaluation_set(data_path, model_dir)
    champ = score(joblib.load(model_dir / MODEL_FILE), X_eval, y_eval)
    chall = score(joblib.load(candidate_dir / MODEL_FILE), X_eval, y_eval)
    promote_it, reasons = decide(champ, chall, len(X_eval))

    if promote_it and dry_run:
        decision = "would promote (dry run, nothing replaced)"
    elif promote_it:
        archive = promote(candidate_dir, model_dir, stamp)
        decision = "promoted"
        reasons.append(f"previous champion archived in {archive.relative_to(model_dir.parent)}")
    else:
        decision = "champion kept"

    entry = {
        "time": stamp,
        "data_file": challenger_meta["data_file"],
        "decision": decision,
        "reasons": reasons,
        "eval_rows": len(X_eval),
        "eval_rows_removed_seen_by_champion": n_removed,
        "champion": {"trained_at": champion_meta["trained_at"], **{k: champ[k] for k in ("mape_pct", "mae_eur", "coverage")}},
        "challenger": {"trained_at": challenger_meta["trained_at"], "folder": str(candidate_dir.relative_to(model_dir.parent)),
                       **{k: chall[k] for k in ("mape_pct", "mae_eur", "coverage")}},
        "challenger_mlflow_run_id": challenger_meta.get("mlflow_run_id"),
    }
    with open(model_dir / "promotion_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    if track and challenger_meta.get("mlflow_run_id"):
        from src.tracking import tag_run

        tag_run(challenger_meta["mlflow_run_id"],
                {"decision": decision, "reason": " | ".join(reasons)},
                {"gate_eval_rows": len(X_eval), "gate_champion_mape_pct": champ["mape_pct"],
                 "gate_challenger_mape_pct": chall["mape_pct"], "gate_challenger_coverage": chall["coverage"]})

    print(f"\nEvaluation rows: {len(X_eval):,} (removed {n_removed:,} the champion was trained on)")
    print(f"Champion:   MAPE {champ['mape_pct']:.3f}%  coverage {champ['coverage']:.1%}")
    print(f"Challenger: MAPE {chall['mape_pct']:.3f}%  coverage {chall['coverage']:.1%}")
    print(f"DECISION: {decision.upper()}")
    for r in reasons:
        print("  -", r)
    if decision == "promoted" and model_dir.resolve() == MODEL_DIR.resolve():
        from src import report

        report.main()  # keep the README results in step with the new live model
        print("README results refreshed. Commit models/ and README.md to publish the new model.")
    return entry


def main():
    parser = argparse.ArgumentParser(description="Retrain on new data and promote only if better.")
    parser.add_argument("--data", required=True, help="CSV with the new data (same columns as the training data)")
    parser.add_argument("--tune", action="store_true", help="full model search instead of the champion's settings")
    parser.add_argument("--dry-run", action="store_true", help="decide, but never replace the live model")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="folder of the live (champion) model")
    parser.add_argument("--no-mlflow", action="store_true", help="skip experiment tracking")
    args = parser.parse_args()
    retrain(args.data, args.model_dir, tune=args.tune, dry_run=args.dry_run, track=not args.no_mlflow)


if __name__ == "__main__":
    main()
