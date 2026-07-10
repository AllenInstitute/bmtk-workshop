import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BMTK_ROOT = REPO_ROOT / "bmtk-dpointnet"
if str(BMTK_ROOT) not in sys.path:
    sys.path.insert(0, str(BMTK_ROOT))

from bmtk.simulator import dpointnet

import tutorial_losses


def run(config_file):
    start = time.perf_counter()
    config = dpointnet.Config.from_json(str(config_file))
    config.build_env()
    rnn = dpointnet.RNN.from_config(config)
    try:
        rnn.run()
    finally:
        rnn.cleanup()
        dpointnet.cleanup_tensorflow()
    elapsed = time.perf_counter() - start
    print(f"Finished {config_file} in {elapsed / 60.0:.2f} min")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_dpointnet.py <config.json>")
    run(sys.argv[1])