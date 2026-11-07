"""Compare instruction gains on the same validation scenes and noise."""
from __future__ import annotations

import math
from pathlib import Path

from models.checkpoints import checkpoint_path

from ._probe import Probe, arguments


def main() -> None:
    parser = arguments(__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gains", nargs="+", type=float, default=[1.0, 1.5])
    args = parser.parse_args()
    if any(not math.isfinite(gain) or gain < 0 for gain in args.gains):
        parser.error("instruction gains must be finite and nonnegative")
    try:
        probe = Probe(args)
        model = probe.load(args.checkpoint or checkpoint_path(probe.cfg, "connector"))
        variants = {str(gain): probe.score(model, gain=gain) for gain in args.gains}
        print(f"Report: {probe.save('probe_gain', variants)}")
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
