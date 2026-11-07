"""Compare named connector checkpoints on matched validation scenes."""
from __future__ import annotations

from pathlib import Path

from models.checkpoints import checkpoint_path

from ._probe import Probe, arguments


def main() -> None:
    parser = arguments(__doc__)
    parser.add_argument("--checkpoint", action="append", metavar="NAME=PATH", help="repeat to compare variants")
    args = parser.parse_args()
    try:
        probe = Probe(args)
        variants = {}
        for item in args.checkpoint or [f"frozen={checkpoint_path(probe.cfg, 'connector')}"]:
            name, separator, path = item.partition("=")
            if not separator or not name or not path or name in variants:
                raise ValueError("--checkpoint expects unique NAME=PATH entries")
            variants[name] = probe.score(probe.load(Path(path)))
        print(f"Report: {probe.save('probe_connector', variants)}")
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
