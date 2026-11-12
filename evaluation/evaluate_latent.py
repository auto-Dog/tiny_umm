"""Compatibility imports for the Tiny Emu3 and BAGEL evaluation entry point."""
from .evaluate_emu3_bagel import (
    Emu3BagelPipeline,
    calibrate_sampling,
    evaluate,
    main,
    sample_attributes,
    save_samples,
)

LatentPipeline = Emu3BagelPipeline


if __name__ == "__main__":
    main()
