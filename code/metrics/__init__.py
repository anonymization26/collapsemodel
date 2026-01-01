"""
Collapse Model core metrics module

Modules:
    reff                — r_eff effective rank computation and Collapse Model prediction formulas
    subspace_alignment  — SA_k subspace alignment (V channel alpha)
    grassmann           — Grassmann manifold geometry tools (principal angles, geodesic distance)
    direction_gain      — Δ_dir direction gain criterion and greedy dataset selection
"""
from .subspace_alignment import compute_subspace_alignment, compute_multiscale_sa
from .reff import (effective_rank, predict_reff_collapse,
                   reff_bounds, reff_theoretical_prediction)
from .grassmann import grassmann_distance, principal_angles, GrassmannLoss
from .direction_gain import direction_gain
