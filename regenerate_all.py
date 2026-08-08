"""Compatibility entry point for the renamed visualization runner.

Prefer ``python generate_visualizations.py`` for new usage.
"""

from generate_visualizations import generate_visualizations, main
from sim.visualize import plot_cherenkov_pool, plot_shower, plot_shower_3d


__all__ = [
    'generate_visualizations',
    'main',
    'plot_cherenkov_pool',
    'plot_shower',
    'plot_shower_3d',
]


if __name__ == '__main__':
    main()
