"""Compatibility imports for the Tiny Emu3 and BAGEL training entry point."""
from .train_emu3_bagel import (
    codec_batch,
    codec_features,
    main,
    train_codec,
    train_model,
    validate_codec,
)


if __name__ == "__main__":
    main()
