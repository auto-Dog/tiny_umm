"""Score a single Flow checkpoint on the validation split."""
from __future__ import annotations

from argparse import BooleanOptionalAction
from pathlib import Path

from models.checkpoints import checkpoint_path

from ._probe import Probe, arguments


def main() -> None:
    parser = arguments(__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--direct-source", action=BooleanOptionalAction, default=None)
    args = parser.parse_args()
    try:
        probe = Probe(args)
        model = probe.load(args.checkpoint or checkpoint_path(probe.cfg, "flow"))
        path = probe.save("probe_flow", {model.metadata["kind"]: probe.score(model, direct=args.direct_source)})
        print(f"Report: {path}")
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
