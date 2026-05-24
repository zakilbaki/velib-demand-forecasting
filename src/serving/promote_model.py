import argparse
import shutil

from src.modeling_config import SERVING_CANDIDATES_DIR, SERVING_MODEL_DIR


def promote_candidate(run_id: str) -> None:
    candidate_dir = SERVING_CANDIDATES_DIR / run_id

    if not candidate_dir.exists():
        raise RuntimeError(
            f"Candidate model not found for run_id={run_id}. "
            "Run src.training.evaluate_regression first."
        )

    if SERVING_MODEL_DIR.exists():
        shutil.rmtree(SERVING_MODEL_DIR)

    shutil.copytree(candidate_dir, SERVING_MODEL_DIR)

    print(f"Promoted candidate {run_id} to {SERVING_MODEL_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a serving model candidate to the current model."
    )
    parser.add_argument("--run-id", required=True, help="MLflow run id of the candidate")
    args = parser.parse_args()

    promote_candidate(args.run_id)


if __name__ == "__main__":
    main()
