"""Factory for the MPE2 spread scenario."""

from ..adapters.mpe import make_mpe


def make_spread(render_mode=None, **kwargs):
    return make_mpe("spread", render_mode=render_mode, **kwargs)
