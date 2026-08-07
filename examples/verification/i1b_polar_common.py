"""Shared helpers for I1-B spherical-polar verification decks."""

CM_PER_UM = 1.0e-4
NS = 1.0e-9

DEFAULT_R_OUTER_CM = 700.0 * CM_PER_UM
DEFAULT_NS_CELLS = 64
DEFAULT_NTHETA_CELLS = 128
DEFAULT_GAMMA = 5.0 / 3.0
DEFAULT_RHO0_GCC = 0.25
DEFAULT_T0_EV = 1.0
DEFAULT_P0_DYN_CM2 = 1.0e14
DEFAULT_T_RAMP_UP_S = 0.5 * NS
DEFAULT_T_HOLD_END_S = 2.5 * NS
DEFAULT_T_RAMP_DOWN_S = 0.5 * NS
DEFAULT_T_END_S = 4.4 * NS
DEFAULT_C_STAR = 8.0

COMPRESSION_METRIC_NOTE = (
    "For no-ambient spherical-polar I1-B decks, compression should be measured "
    "from the body-fitted outer s-boundary or an equivalent volume radius; "
    "Stage-1 a0=0 uses a single gamma-law gas and has no material interface."
)


def c2_smoothstep(x):
    """Clamped C^2 smoothstep S(x)=10x^3-15x^4+6x^5."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def ablation_pressure_ramp(t, p0, t_r, t_h, t_d):
    """C^2 pressure ramp with absolute hold-end time t_h."""
    if t <= 0.0:
        return 0.0
    if t <= t_r:
        return p0 * c2_smoothstep(t / t_r)
    if t <= t_h:
        return p0
    if t <= t_h + t_d:
        return p0 * (1.0 - c2_smoothstep((t - t_h) / t_d))
    return 0.0


def make_gamma_law_material(Material, name="gas", gamma=DEFAULT_GAMMA, A=1.0, Z=1.0):
    """Build the single ideal gamma-law material used by polar hydro decks."""
    return Material(
        name=name,
        A=A,
        Z=Z,
        eos=dict(model="ideal_gas", ideal_gas=dict(gamma=gamma)),
        opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
    )
