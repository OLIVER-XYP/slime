"""Optional TCO evidence hooks for Slime's RL pipeline."""

from .observer import SlimeTCOObserver, get_global_observer

__all__ = ["SlimeTCOObserver", "get_global_observer"]
