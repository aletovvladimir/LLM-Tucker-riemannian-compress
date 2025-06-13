import subprocess
from pathlib import Path

import hydra


@hydra.main(config_path="../configs", config_name="config")
def run_mlflow_ui(config):
    base_dir = Path(__file__).resolve().parents[2]

    mlruns_path = base_dir / "plots" / "mlruns"

    if not mlruns_path.exists():
        print(f"No such directory: {mlruns_path}")
        return

    print(f"Run MLFlow from: {mlruns_path}")
    subprocess.run(
        [
            "mlflow",
            "ui",
            "--backend-store-uri",
            f"file:{mlruns_path}",
            "--port",
            config.mlflow.port,
        ]
    )


if __name__ == "__main__":
    run_mlflow_ui()
