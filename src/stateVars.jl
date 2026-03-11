"""
stateVars.jl

Defines the core GPU-resident State container for the
two-layer rotating shallow-water model.
"""

# ============================================================
# Prognostic variables (time-stepped state)
# ============================================================

struct Prognostic
    h::CuArray{FT,2}   # thickness (HGRID)
    m::CuArray{FT,2}   # h*u (UGRID)
    n::CuArray{FT,2}   # h*v (VGRID)
end


# ============================================================
# Intermediate variables (predictor stage)
# ============================================================

struct Intermediate
    h_star::CuArray{FT,2}
    m_star::CuArray{FT,2}
    n_star::CuArray{FT,2}
end


# ============================================================
# Diagnostic variables (recomputed every RHS)
# ============================================================

struct Diagnostic
    u::CuArray{FT,2}
    gradx_u::CuArray{FT,2}
    grady_u::CuArray{FT,2}
    v::CuArray{FT,2}
    gradx_v::CuArray{FT,2}
    grady_v::CuArray{FT,2}
    zeta::CuArray{FT,2}

    advec_term_u::CuArray{FT,2}
    advec_term_v::CuArray{FT,2}

    pressure::CuArray{FT,2}
    gradx_pres::CuArray{FT,2}
    grady_pres::CuArray{FT,2}

    viscosity_u::CuArray{FT,2}
    viscosity_v::CuArray{FT,2}
    viscous_term_u::CuArray{FT,2}
    viscous_term_v::CuArray{FT,2}

    advx_curv::CuArray{FT,2}
    advy_curv::CuArray{FT,2}
    visx_curv::CuArray{FT,2}
    visy_curv::CuArray{FT,2}

    coriolis_term_u::CuArray{FT,2}
    coriolis_term_v::CuArray{FT,2}
end


# ============================================================
# AB3–AM3 history storage
# ============================================================

struct History
    rhs_m_tp1::CuArray{FT,2}
    rhs_m_tm0::CuArray{FT,2}
    rhs_m_tm1::CuArray{FT,2}
    rhs_m_tm2::CuArray{FT,2}

    rhs_n_tp1::CuArray{FT,2}
    rhs_n_tm0::CuArray{FT,2}
    rhs_n_tm1::CuArray{FT,2}
    rhs_n_tm2::CuArray{FT,2}

    rhs_h_tp1::CuArray{FT,2}
    rhs_h_tm0::CuArray{FT,2}
    rhs_h_tm1::CuArray{FT,2}
    rhs_h_tm2::CuArray{FT,2}
end


# ============================================================
# Shared interpolated variables
# ============================================================

struct Interpolated
    h_in_u::CuArray{FT,2}
    h_in_v::CuArray{FT,2}
    u_in_v::CuArray{FT,2}
    v_in_u::CuArray{FT,2}
end


# ============================================================
# Forcing variables
# ============================================================

struct Forcing
    uwind::CuArray{FT,2}
    vwind::CuArray{FT,2}

    taux_sf::CuArray{FT,2}
    tauy_sf::CuArray{FT,2}

    taux_bt::CuArray{FT,2}
    tauy_bt::CuArray{FT,2}
end


# ============================================================
# Scratch space (shared temporary arrays)
# ============================================================

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
end


# ============================================================
# Top-level model state
# ============================================================

struct State
    prog1::Prognostic
    prog2::Prognostic
    diag1::Diagnostic
    diag2::Diagnostic
    hist1::History
    hist2::History
    intm1::Intermediate
    intm2::Intermediate
    intp::Interpolated
    forc::Forcing
    temp::Temporary
end


# ============================================================
# Allocation routine
# ============================================================

function allocate_state(p::Params)::State
    Nx, Ny = p.Nx, p.Ny

    # Prognostic
    prog1 = Prognostic(
        CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny),
    )

    prog2 = Prognostic(
        CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny),
    )

    # Intermediate
    intm1 = Intermediate(
        CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny),
    )

    intm2 = Intermediate(
        CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny),
    )

    # History
    hist1 = History(
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
    )

    hist2 = History(
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
    )

    # Interpolated
    intp = Interpolated(
        CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny),
    )

    # Diagnostic
    diag1 = Diagnostic(
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
    )

    diag2 = Diagnostic(
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
    )

    # Forcing
    forc = Forcing(
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
    )

    # Temporary
    temp = Temporary(
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
        CUDA.zeros(FT, Nx, Ny), CUDA.zeros(FT, Nx, Ny),
    )

    return State(
        prog1, prog2,
        diag1, diag2,
        hist1, hist2,
        intm1, intm2,
        intp, forc, temp,
    )
end