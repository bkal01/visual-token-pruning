from enum import Enum

class RoPEConfig(Enum):
    """
    DEFAULT: model keeps whatever RoPE implementation is has.
    NONE: remove RoPE from the model.
    """
    DEFAULT = 1
    NONE = 2

