"""Compatibility imports for the Tiny Emu3 and BAGEL implementations."""
from .emu3_bagel import (
    LatentGrid,
    TinyBagel,
    TinyEmu3,
    dropped_prompt,
    foreground_weights,
    unconditional_prompt,
)

LatentEmu3 = TinyEmu3
LatentBagel = TinyBagel
