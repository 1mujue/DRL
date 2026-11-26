"""Factory for the MPE2 tag scenario."""

from ..adapters.mpe import make_mpe


def make_tag(render_mode=None, **kwargs):
    return make_mpe("tag", render_mode=render_mode, **kwargs)
