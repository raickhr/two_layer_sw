"""
stateVars.jl

Defines the core **State container** for the two-layer rotating shallow-water model
and allocates all GPU-resident arrays.

This file centralizes *all mutable model fields* so that:
- No CuArray allocations occur inside time-stepping routines
- Barotropic and baroclinic solvers reuse shared scratch buffers
- Memory layout and ownership are explicit and easy to reason about

Assumptions
-----------
- `FT` is the floating-point type (e.g. Float32 or Float64), defined elsewhere
- `Params` contains at least:
    * `Nx, Ny :: Int`   : horizontal grid size
    * `H1, H2 :: Real` : resting layer thicknesses
- `CUDA` is available in the parent module (`using CUDA`)
"""


# ============================================================
# Prognostic variables (time-stepped state)
# ============================================================
"""
    Prognostic

Holds all **prognostic (time-integrated)** variables of the model.

Grid staggering:
- Thicknesses (`h*`, `H`) live at **h-points**
- Zonal transports (`m*`, `M`) live at **u-points**
- Meridional transports (`n*`, `N`) live at **v-points**

Fields
------
Layer thicknesses:
- `h1(i,j)` : top-layer thickness
- `h2(i,j)` : bottom-layer thickness
- `H(i,j)`  : total thickness = h1 + h2
- `H_old`   : total thickness at previous *baroclinic* time step
              (used for explicit pressure/forcing splitting)

Zonal mass transports:
- `m1 = h1 * u1`
- `m2 = h2 * u2`
- `M  = H  * U`   (barotropic transport)

Meridional mass transports:
- `n1 = h1 * v1`
- `n2 = h2 * v2`
- `N  = H  * V`

Notes
-----
- All arrays are **mutated in place**
- Time-level separation is handled algorithmically, not by duplicate storage
"""
struct Prognostic
    h1::CuArray{FT,2}
    h2::CuArray{FT,2}
    H::CuArray{FT,2}
    H_old::CuArray{FT,2}

    m1::CuArray{FT,2}
    m2::CuArray{FT,2}
    M::CuArray{FT,2}

    n1::CuArray{FT,2}
    n2::CuArray{FT,2}
    N::CuArray{FT,2}
end


# ============================================================
# External forcing fields
# ============================================================
"""
    Forcing

Holds all externally prescribed forcing fields.

Fields
------
Atmospheric forcing:
- `uwind`, `vwind` : 10 m wind velocity components on the model grid

Stress fields:
- `taux_sf`, `tauy_sf` : surface wind stress
- `taux_bt`, `tauy_bt` : bottom drag stress

Notes
-----
- Stresses are computed in `forcing.jl`
- They are applied in both barotropic and baroclinic solvers
"""
struct Forcing
    uwind::CuArray{FT,2}
    vwind::CuArray{FT,2}

    taux_sf::CuArray{FT,2}
    tauy_sf::CuArray{FT,2}

    taux_bt::CuArray{FT,2}
    tauy_bt::CuArray{FT,2}
end


# ============================================================
# Temporary work arrays (scratch space)
# ============================================================
"""
    Temporary

Reusable **scratch arrays** for intermediate computations.

These buffers are intentionally generic and are **aliased differently**
in different parts of the solver:

- barotropic solver
- baroclinic solver
- reconstructions
- forcing calculations

Design rules
------------
- No kernel allocates memory
- Aliasing is safe because usage phases do not overlap
- Size is always `(Nx, Ny)`
"""
struct Temporary
    temp_var_x1::CuArray{FT,2}
    temp_var_y1::CuArray{FT,2}

    temp_var_x2::CuArray{FT,2}
    temp_var_y2::CuArray{FT,2}

    temp_var_x3::CuArray{FT,2}
    temp_var_y3::CuArray{FT,2}

    temp_var_x4::CuArray{FT,2}
    temp_var_y4::CuArray{FT,2}

    temp_var_x5::CuArray{FT,2}
    temp_var_y5::CuArray{FT,2}

    temp_var_x6::CuArray{FT,2}
    temp_var_y6::CuArray{FT,2}

    temp_var_x7::CuArray{FT,2}
    temp_var_y7::CuArray{FT,2}

    temp_var_x8::CuArray{FT,2}
    temp_var_y8::CuArray{FT,2}

    temp_var_x9::CuArray{FT,2}
    temp_var_y9::CuArray{FT,2}
end


# ============================================================
# Top-level model state
# ============================================================
"""
    State

Top-level container holding **all mutable model state**.

Components
----------
- `prog :: Prognostic` : time-evolving physical state
- `forc :: Forcing`    : externally imposed forcing
- `temp :: Temporary` : shared scratch space

All time-stepping routines mutate a `State` in place.
"""
struct State
    prog::Prognostic
    forc::Forcing
    temp::Temporary
end


# ============================================================
# Allocation routine
# ============================================================
"""
    allocate_state(p::Params) -> State

Allocate and initialize all model fields on the GPU.

Initialization
--------------
- `h1 = H1`, `h2 = H2`, `H = H1 + H2`
- All transports set to zero
- Forcing fields initialized to zero
- All temporary buffers zeroed

No allocations occur after this call.
"""
function allocate_state(p::Params)::State
    Nx, Ny = p.Nx, p.Ny

    # ---------- Prognostic ----------
    h1 = CUDA.fill(FT(p.H1), Nx, Ny)
    h2 = CUDA.fill(FT(p.H2), Nx, Ny)
    H  = CUDA.fill(FT(p.H1 + p.H2), Nx, Ny)
    H_old = CUDA.fill(FT(p.H1 + p.H2), Nx, Ny)

    m1 = CUDA.zeros(FT, Nx, Ny)
    m2 = CUDA.zeros(FT, Nx, Ny)
    M  = CUDA.zeros(FT, Nx, Ny)

    n1 = CUDA.zeros(FT, Nx, Ny)
    n2 = CUDA.zeros(FT, Nx, Ny)
    N  = CUDA.zeros(FT, Nx, Ny)

    prog = Prognostic(h1, h2, H, H_old, m1, m2, M, n1, n2, N)

    # ---------- Forcing ----------
    uwind = CUDA.zeros(FT, Nx, Ny)
    vwind = CUDA.zeros(FT, Nx, Ny)

    taux_sf = CUDA.zeros(FT, Nx, Ny)
    tauy_sf = CUDA.zeros(FT, Nx, Ny)
    taux_bt = CUDA.zeros(FT, Nx, Ny)
    tauy_bt = CUDA.zeros(FT, Nx, Ny)

    forc = Forcing(uwind, vwind, taux_sf, tauy_sf, taux_bt, tauy_bt)

    # ---------- Temporary ----------
    temp = Temporary(
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
    )

    return State(prog, forc, temp)
end
