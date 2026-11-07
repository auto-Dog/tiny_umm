"""Stage C: replace the Stage-B visual encoder with a bridge from understanding."""
from __future__ import annotations

from models.checkpoints import apply_flow_state, checkpoint_path, load_flow, read_checkpoint
from models.tiny_umm import TinyUMM

from .common import arguments, setup
from .train_flow import train_flow_model


def main() -> None:
    parser = arguments(__doc__)
    parser.add_argument("--stage", choices=("c", "discrete", "d"), default="c")
    parser.add_argument("--joint-instruction", action="store_true", help="also train the Stage-B instruction encoder (C only)")
    args = parser.parse_args()
    if args.joint_instruction and args.stage != "c":
        parser.error("--joint-instruction is only valid with --stage c")
    cfg, device = setup(args)
    settings = dict(cfg["training"]["connector"])
    joint = args.stage == "c" and (args.joint_instruction or not settings["freeze_instruction"])
    if args.stage == "d":
        model, saved = load_flow(cfg, checkpoint_path(cfg, "connector"), device)
        kind = "adapter"
        settings.update(cfg["training"]["adapter"])
        settings["direct_source"] = saved["direct_source"]
    else:
        kind = "discrete" if args.stage == "discrete" else ("joint" if joint else "connector")
        model = TinyUMM(cfg, "discrete" if args.stage == "discrete" else "connector")
        flow = read_checkpoint(cfg, checkpoint_path(cfg, "flow"), ("flow",))
        understanding = read_checkpoint(cfg, checkpoint_path(cfg, "understanding"), ("understanding",))
        apply_flow_state(model, flow["state"])
        model.understanding.load_state_dict(understanding["state"])
    model.train_bridge(joint_instruction=joint, adapters=kind == "adapter")
    train_flow_model(
        cfg, device, model, kind, settings, generation_ratio=settings["generation_ratio"],
        joint=joint, reconstruct_epochs=settings["reconstruct_epochs"] if args.stage == "c" else 0,
    )


if __name__ == "__main__":
    main()
