# baroclinic.jl
#
# Routines for advancing the **baroclinic (internal) modes** of the 2-layer
# rotating shallow-water system on a C-grid.
#
# Time stepping:
#   - Explicit RK4 (4th-order Runge–Kutta) for **non-Coriolis** terms:
#       * baroclinic pressure / form drag
#       * curvature / metric terms
#       * surface & bottom stress splitting
#       * biharmonic viscosity (∝ ∇⁴) on (m, n)
#       * advective transport of (m, n) and h (WENO-Z, flux form)
#   - Semi-implicit Coriolis update with existing `k_add_coriolisforce!`
#   - Shapiro filter on h (10x weaker than barotropic)
#
# No CuArray allocations occur here; all scratch space is taken from
# `state.temp::Temporary`.
#
# Main entry points
# -----------------
# - `step_baroclinic_layer!(...)` : RK4 step for a single layer (1 or 2)
# - `step_baroclinic!(...)`       : convenience wrapper (layer 1 then layer 2)


# ============================================================
# Non-Coriolis baroclinic RHS
# ============================================================

"""
    compute_baroclinic_rhs_nonCoriolis!(
        k_m, k_n, k_h,
        m_state, n_state, h_state,
        state, grid, p, layer,
        threads1, blocks1, threads2, blocks2;
        h_in_u, h_in_v, buf_x, buf_y, termx, termy, u_face, v_face,
        mode_split,
    )

Compute **non-Coriolis** baroclinic tendencies for a given stage state
`(m_state, n_state, h_state)` of a single layer:

    k_m = (form drag + baroclinic PGF + curvature + forcing
           + biharmonic viscosity + advection of m_state)

    k_n = same for meridional transport

    k_h = (-∇·(u h_state))

All arguments are CuArray{FT,2} with size (Nx,Ny).

This routine does **not** apply Coriolis or Shapiro filtering; those are
handled around the RK4 driver.
"""
function compute_baroclinic_rhs_nonCoriolis!(
    k_m, k_n, k_h,
    m_state, n_state, h_state,
    state::State,
    grid::Grid,
    p::Params,
    layer::Int,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int};
    h_in_u,
    h_in_v,
    buf_x,
    buf_y,
    termx,
    termy,
    u_face,
    v_face,
    mode_split::Bool,
)
    Nx, Ny = p.Nx, p.Ny

    # -------------------------
    # Grid metrics
    # -------------------------
    dx_n2n_h  = grid.dx_n2n_h
    dy_n2n_h  = grid.dy_n2n_h
    dx_face_h = grid.dx_face_h
    dy_face_h = grid.dy_face_h
    dArea_h   = grid.dArea_h

    dx_n2n_u  = grid.dx_n2n_u
    dy_n2n_u  = grid.dy_n2n_u
    dx_face_u = grid.dx_face_u
    dy_face_u = grid.dy_face_u
    dArea_u   = grid.dArea_u

    dx_n2n_v  = grid.dx_n2n_v
    dy_n2n_v  = grid.dy_n2n_v
    dx_face_v = grid.dx_face_v
    dy_face_v = grid.dy_face_v
    dArea_v   = grid.dArea_v

    lat_u     = grid.lat_u
    lat_v     = grid.lat_v

    # -------------------------
    # Scalars
    # -------------------------
    g      = FT(p.g)
    gp     = FT(p.gp)
    ρ1     = FT(p.rho1)
    ρ2     = FT(p.rho2)
    hmin   = FT(p.hmin)
    ν      = FT(p.nu)
    Rearth = FT(p.earthRadius)

    # -------------------------
    # Aliases to prognostic / forcing
    # -------------------------
    prog   = state.prog
    forc   = state.forc

    h1     = prog.h1
    h2     = prog.h2
    H_old  = prog.H_old

    taux_sf = forc.taux_sf
    tauy_sf = forc.tauy_sf
    taux_bt = forc.taux_bt
    tauy_bt = forc.tauy_bt

    # Scratch aliases
    h1_in_u = buf_x   # first use of buf_x / buf_y: h1 reconstructions
    h1_in_v = buf_y

    # Ensure tendencies start from zero
    @. k_m = FT(0)
    @. k_n = FT(0)
    @. k_h = FT(0)

    # ========================================================
    # 1. Wall BCs for meridional transport (no-normal flow)
    # ========================================================
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n_state, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h1, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h_state, Nx, Ny)

    # ========================================================
    # 2. Reconstructions: h₁ at u/v, active layer h at u/v
    # ========================================================
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h1_in_u, h1, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h1_in_v, h1, Nx, Ny, hmin)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h_in_u, h_state, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h_in_v, h_state, Nx, Ny, hmin)

    # ========================================================
    # 3. Form drag / baroclinic pressure gradient
    #    based on ∇h₂ and layer-dependent coefficients.
    # ========================================================

    # termx, termy: ∂h₂/∂x, ∂h₂/∂y at h-points
    @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
        termx, termy,
        h2,
        dx_n2n_h, dy_n2n_h,
        Nx, Ny
    )

    if layer == 1
        # + g * h₁ ∇h₂
        @. termx = termx * (g * h1_in_u)
        @. termy = termy * (g * h1_in_v)
    else
        # (- g h₁ - g' h₂) ∇h₂
        @. termx = termx * (-g * h1_in_u - gp * h_in_u)
        @. termy = termy * (-g * h1_in_v - gp * h_in_v)
    end

    # Always add baroclinic contribution
    @. k_m = k_m + termx
    @. k_n = k_n + termy
    
    # If not splitting, also include barotropic PGF from total thickness H
    if !mode_split
        @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
            termx, termy,
            prog.H,              # total thickness at h-points
            dx_n2n_h, dy_n2n_h,
            Nx, Ny
        )

        # add g h ∇H term
        @. termx = (g * h_in_u) * termx
        @. termy = (g * h_in_v) * termy

        @. k_m = k_m + termx
        @. k_n = k_n + termy
    end

    # ========================================================
    # 4. Curvature / metric terms
    # ========================================================
    @cuda threads=threads2 blocks=blocks2 k_calc_curvature_terms!(
        termx, termy,
        m_state, n_state,
        h_in_u, h_in_v,
        lat_u, lat_v,
        Nx, Ny,
        Rearth
    )

    @. k_m = k_m + termx
    @. k_n = k_n + termy

    # ========================================================
    # 5. Baroclinic forcing split using H_old
    # ========================================================

    if mode_split
        H_in_u = buf_x   # reuse buf_x / buf_y
        H_in_v = buf_y

        @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(H_old, Nx, Ny)        
        @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(H_in_u, H_old, Nx, Ny, hmin)
        @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(H_in_v, H_old, Nx, Ny, hmin)

        if layer == 1
            @. termx = taux_sf/ρ1 - (h_in_u / H_in_u) * (taux_sf/ρ1 + taux_bt/ρ2)
            @. termy = tauy_sf/ρ1 - (h_in_v / H_in_v) * (tauy_sf/ρ1 + tauy_bt/ρ2)
        else
            @. termx = taux_bt/ρ2 - (h_in_u / H_in_u) * (taux_sf/ρ1 + taux_bt/ρ2)
            @. termy = tauy_bt/ρ2 - (h_in_v / H_in_v) * (tauy_sf/ρ1 + tauy_bt/ρ2)
        end
    else
        if layer == 1
            @. termx = taux_sf/ρ1
            @. termy = tauy_sf/ρ1
        else
            @. termx = taux_bt/ρ2 
            @. termy = tauy_bt/ρ2
        end
    end

    @. k_m = k_m + termx
    @. k_n = k_n + termy

    # ========================================================
    # 6. Viscosity: biharmonic on transports (m_state, n_state)
    #    We interpret p.nu as sqrt(A4), so:
    #      L(q)   = ∇·(ν ∇q)  ≈ ν ∇² q
    #      L(L(q)) ≈ ν² ∇⁴ q
    #    Add dissipative term:  -ν² ∇⁴ q
    # ========================================================
    gradx = buf_x  # reuse buf_x / buf_y as gradient buffers
    grady = buf_y

    # For m_state (u-grid)
    @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
        gradx, grady,
        m_state,
        dx_n2n_u, dy_n2n_u,
        Nx, Ny
    )

    @cuda threads=threads2 blocks=blocks2 k_calc_laplacian!(
        termx, gradx, grady,
        dx_face_u, dy_face_u,
        dArea_u,
        ν,
        Nx, Ny
    )

    @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
        gradx, grady,
        termx,
        dx_n2n_u, dy_n2n_u,
        Nx, Ny
    )

    @cuda threads=threads2 blocks=blocks2 k_calc_laplacian!(
        termx, gradx, grady,
        dx_face_u, dy_face_u,
        dArea_u,
        ν,
        Nx, Ny
    )

    @. k_m = k_m - termx

    # For n_state (v-grid)
    @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
        gradx, grady,
        n_state,
        dx_n2n_v, dy_n2n_v,
        Nx, Ny
    )

    @cuda threads=threads2 blocks=blocks2 k_calc_laplacian!(
        termy, gradx, grady,
        dx_face_v, dy_face_v,
        dArea_v,
        ν,
        Nx, Ny
    )

    @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
        gradx, grady,
        termy,
        dx_n2n_v, dy_n2n_v,
        Nx, Ny
    )

    @cuda threads=threads2 blocks=blocks2 k_calc_laplacian!(
        termy, gradx, grady,
        dx_face_v, dy_face_v,
        dArea_v,
        ν,
        Nx, Ny
    )

    @. k_n = k_n - termy

    # ========================================================
    # 7. Advective transport of (m_state, n_state)
    #    WENO-Z, flux-form, on u- and v-cells.
    # ========================================================
    u_face_for_u = u_face
    v_face_for_u = v_face

    # u-cells (advecting m_state)
    @cuda threads=threads2 blocks=blocks2 k_calc_faceVels_for_ucell!(
        u_face_for_u, v_face_for_u,
        m_state, n_state,
        h_state,
        Nx, Ny,
        hmin
    )

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(v_face_for_u, Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        termx,
        m_state,
        u_face_for_u, v_face_for_u,
        dx_face_u, dy_face_u,
        dArea_u,
        Nx, Ny
    )

    @. k_m = k_m + termx

    # v-cells (advecting n_state)
    u_face_for_v = u_face
    v_face_for_v = v_face

    @cuda threads=threads2 blocks=blocks2 k_calc_faceVels_for_vcell!(
        u_face_for_v, v_face_for_v,
        m_state, n_state,
        h_state,
        Nx, Ny,
        hmin
    )

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(v_face_for_v, Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        termy,
        n_state,
        u_face_for_v, v_face_for_v,
        dx_face_v, dy_face_v,
        dArea_v,
        Nx, Ny
    )

    @. k_n = k_n + termy

    # ========================================================
    # 8. Mass equation: thickness advection of h_state
    #    k_h = ( -∇·(u h_state) )
    # ========================================================
    # Reuse u_face, v_face as h-cell face velocities
    @. u_face = m_state / h_in_u
    @. v_face = n_state / h_in_v

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(v_face, Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        k_h,
        h_state,
        u_face, v_face,
        dx_face_h, dy_face_h,
        dArea_h,
        Nx, Ny
    )

    # k_h already holds (-∇·(u h))
    return nothing
end


# ============================================================
# Single RK4 baroclinic step for one layer
# ============================================================

"""
    rk4_step_baroclinic_layer_once!(
        state::State,
        grid::Grid,
        p::Params,
        dt::FT,
        layer::Int;
        threads1, blocks1, threads2, blocks2,
    )

Perform one RK4 time step of size `dt` for baroclinic variables
(m_layer, n_layer, h_layer) of the specified `layer` (1 or 2):

- Uses `compute_baroclinic_rhs_nonCoriolis!` for all non-Coriolis terms.
- Applies one semi-implicit Coriolis update via `k_add_coriolisforce!`.
- Applies a weak Shapiro filter to h (0.1 × p.smoothing_eps).

Results are written back into:
- `prog.m1, prog.n1, prog.h1` for `layer == 1`
- `prog.m2, prog.n2, prog.h2` for `layer == 2`
"""
function rk4_step_baroclinic_layer_once!(
    state::State,
    grid::Grid,
    p::Params,
    dt::FT,
    layer::Int;
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
    mode_split::Bool,
)
    Nx, Ny = p.Nx, p.Ny

    prog = state.prog

    # Select layer prognostics
    m = (layer == 1) ? prog.m1 : prog.m2
    n = (layer == 1) ? prog.n1 : prog.n2
    h = (layer == 1) ? prog.h1 : prog.h2

    # Scalars
    Ω        = FT(p.Ω)
    smoothϵ  = FT(p.smoothing_eps) * FT(0.1)  # weaker for baroclinic
    lat_u    = grid.lat_u
    lat_v    = grid.lat_v

    # RK4 weights
    dt_sixth = dt / FT(6)
    dt_third = dt / FT(3)
    dt_half  = dt / FT(2)

    temp = state.temp

    # --------------------------------------------------------
    # Alias Temporary arrays for RK4 and scratch (no allocs)
    # --------------------------------------------------------

    # Accumulators: q_accum = q_n + Σ(weights * k)
    m_accum = temp.temp_var_x1
    n_accum = temp.temp_var_y1
    h_accum = temp.temp_var_x2

    # Stage states
    m_stage = temp.temp_var_y2
    n_stage = temp.temp_var_x3
    h_stage = temp.temp_var_y3

    # Current stage tendencies
    k_m = temp.temp_var_x4
    k_n = temp.temp_var_y4
    k_h = temp.temp_var_x5

    # Scratch for RHS
    h_in_u = temp.temp_var_x6
    h_in_v = temp.temp_var_y5
    buf_x  = temp.temp_var_x7
    buf_y  = temp.temp_var_y6
    termx  = temp.temp_var_x8
    termy  = temp.temp_var_y7
    u_face = temp.temp_var_x9
    v_face = temp.temp_var_y8
    # temp.temp_var_y9 remains free if needed later

    # -------------------------
    # Initialize accumulators
    # -------------------------
    @. m_accum = m
    @. n_accum = n
    @. h_accum = h

    # ========================================================
    # Stage 1: k1 = f(q_n)
    # ========================================================
    @. m_stage = m
    @. n_stage = n
    @. h_stage = h

    compute_baroclinic_rhs_nonCoriolis!(
        k_m, k_n, k_h,
        m_stage, n_stage, h_stage,
        state, grid, p, layer,
        threads1, blocks1, threads2, blocks2;
        h_in_u=h_in_u,
        h_in_v=h_in_v,
        buf_x=buf_x,
        buf_y=buf_y,
        termx=termx,
        termy=termy,
        u_face=u_face,
        v_face=v_face,
        mode_split=mode_split,
    )

    # Accumulate dt/6 * k1
    @. m_accum = m_accum + dt_sixth * k_m
    @. n_accum = n_accum + dt_sixth * k_n
    @. h_accum = h_accum + dt_sixth * k_h

    # Stage state for k2: q_stage = q_n + dt/2 * k1
    @. m_stage = m + dt_half * k_m
    @. n_stage = n + dt_half * k_n
    @. h_stage = h + dt_half * k_h

    # ========================================================
    # Stage 2: k2 = f(q_n + dt/2 * k1)
    # ========================================================
    compute_baroclinic_rhs_nonCoriolis!(
        k_m, k_n, k_h,
        m_stage, n_stage, h_stage,
        state, grid, p, layer,
        threads1, blocks1, threads2, blocks2;
        h_in_u=h_in_u,
        h_in_v=h_in_v,
        buf_x=buf_x,
        buf_y=buf_y,
        termx=termx,
        termy=termy,
        u_face=u_face,
        v_face=v_face,
        mode_split=mode_split,
    )

    # Accumulate dt/3 * k2
    @. m_accum = m_accum + dt_third * k_m
    @. n_accum = n_accum + dt_third * k_n
    @. h_accum = h_accum + dt_third * k_h

    # Stage state for k3: q_stage = q_n + dt/2 * k2
    @. m_stage = m + dt_half * k_m
    @. n_stage = n + dt_half * k_n
    @. h_stage = h + dt_half * k_h

    # ========================================================
    # Stage 3: k3 = f(q_n + dt/2 * k2)
    # ========================================================
    compute_baroclinic_rhs_nonCoriolis!(
        k_m, k_n, k_h,
        m_stage, n_stage, h_stage,
        state, grid, p, layer,
        threads1, blocks1, threads2, blocks2;
        h_in_u=h_in_u,
        h_in_v=h_in_v,
        buf_x=buf_x,
        buf_y=buf_y,
        termx=termx,
        termy=termy,
        u_face=u_face,
        v_face=v_face,
        mode_split=mode_split,
    )

    # Accumulate dt/3 * k3
    @. m_accum = m_accum + dt_third * k_m
    @. n_accum = n_accum + dt_third * k_n
    @. h_accum = h_accum + dt_third * k_h

    # Stage state for k4: q_stage = q_n + dt * k3
    @. m_stage = m + dt * k_m
    @. n_stage = n + dt * k_n
    @. h_stage = h + dt * k_h

    # ========================================================
    # Stage 4: k4 = f(q_n + dt * k3)
    # ========================================================
    compute_baroclinic_rhs_nonCoriolis!(
        k_m, k_n, k_h,
        m_stage, n_stage, h_stage,
        state, grid, p, layer,
        threads1, blocks1, threads2, blocks2;
        h_in_u=h_in_u,
        h_in_v=h_in_v,
        buf_x=buf_x,
        buf_y=buf_y,
        termx=termx,
        termy=termy,
        u_face=u_face,
        v_face=v_face,
        mode_split=mode_split,
    )

    # Accumulate dt/6 * k4
    @. m_accum = m_accum + dt_sixth * k_m
    @. n_accum = n_accum + dt_sixth * k_n
    @. h_accum = h_accum + dt_sixth * k_h

    # -------------------------
    # Write back explicit RK4 result
    # -------------------------
    m .= m_accum
    n .= n_accum
    h .= h_accum

    # ========================================================
    # Semi-implicit Coriolis + Shapiro filter on h
    # ========================================================
    # Reuse some temp arrays for Coriolis / Shapiro:
    m_old  = k_m          # temp_var_x4
    n_old  = k_n          # temp_var_y4
    h_star = k_h          # temp_var_x5

    # Copy current m,n into m_old,n_old
    m_old .= m
    n_old .= n

    # Use m_stage, n_stage as "m_star, n_star" inputs
    m_star = m_stage     # temp_var_y2
    n_star = n_stage     # temp_var_x3

    m_star .= m
    n_star .= n

    @cuda threads=threads2 blocks=blocks2 k_add_coriolisforce!(
        m, n,
        m_old, n_old,
        m_star, n_star,
        lat_u, lat_v,
        Nx, Ny,
        dt, Ω
    )

    # Shapiro smoothing on thickness h
    h_star .= h

    @cuda threads=threads2 blocks=blocks2 k_apply_shapiro_filter!(
        h,
        h_star,
        smoothϵ,
        Nx, Ny
    )

    return nothing
end


# ============================================================
# Public API: baroclinic steps
# ============================================================

"""
    step_baroclinic_layer!(
        state::State,
        grid::Grid,
        p::Params,
        threads1::Int,
        blocks1::Int,
        threads2::NTuple{2,Int},
        blocks2::NTuple{2,Int};
        layer::Int = 1,
    )

Advance a **single baroclinic layer** (1 or 2) by one baroclinic time step
`p.dt` using RK4 for non-Coriolis terms, semi-implicit Coriolis, and
weak Shapiro filtering on thickness.
"""
function step_baroclinic_layer!(
    state::State,
    grid::Grid,
    p::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int};
    layer::Int = 1,
    mode_split::Bool = true,
)
    dt = FT(p.dt)

    rk4_step_baroclinic_layer_once!(
        state,
        grid,
        p,
        dt,
        layer;
        threads1=threads1,
        blocks1=blocks1,
        threads2=threads2,
        blocks2=blocks2,
        mode_split,
    )

    return nothing
end


"""
    step_baroclinic!(
        state::State,
        grid::Grid,
        p::Params;
        threads1::Int,
        blocks1::Int,
        threads2::NTuple{2,Int},
        blocks2::NTuple{2,Int},
        mode_split::Bool = true,
    )

Advance both baroclinic layers (1 then 2) by one baroclinic time step `p.dt`.
"""
function step_baroclinic!(
    state::State,
    grid::Grid,
    p::Params;
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
    mode_split::Bool = true,
)
    step_baroclinic_layer!(
        state, grid, p,
        threads1, blocks1, threads2, blocks2;
        layer=1, mode_split
    )
    step_baroclinic_layer!(
        state, grid, p,
        threads1, blocks1, threads2, blocks2;
        layer=2, mode_split
    )
    return nothing
end
