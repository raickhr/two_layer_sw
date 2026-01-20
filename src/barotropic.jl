# barotropic.jl
#
# Barotropic (external-mode) RK4 subcycling for the
# two-layer rotating shallow-water model.
#
# This module advances the fast external mode (M, N, H) by performing
# p.M barotropic substeps of size p.dtBT for each baroclinic timestep.
#
# Time stepping:
#   - Explicit RK4 for non-Coriolis terms:
#       * barotropic pressure gradients
#       * curvature (metric) terms
#       * wind + bottom stresses
#       * mass divergence (WENO-Z flux-form)
#   - Semi-implicit Coriolis rotation (k_add_coriolisforce!)
#   - Shapiro filter on H after each substep
#
# All scratch arrays come from `state.temp` (no allocations here).


# ============================================================
# Non-Coriolis barotropic RHS
# ============================================================

"""
    compute_barotropic_rhs_nonCoriolis!(
        dM, dN, dH,
        M, N, H,
        state, grid, p,
        threads1, blocks1, threads2, blocks2;
        H_in_u, H_in_v,
        termx, termy,
        U_face, V_face,
    )

Compute the **non-Coriolis** barotropic tendencies for a given
barotropic state (M, N, H):

    dM = (pressure + curvature + forcing)_x
    dN = (pressure + curvature + forcing)_y
    dH = -∇·(U H)

All arrays are CuArray{FT,2} of size (Nx,Ny).

Scratch arrays:
- `H_in_u, H_in_v` : thickness reconstructed to u/v points
- `termx, termy`   : reused as ∂H/∂x, ∂H/∂y, curvature_x, curvature_y, forcing_x, forcing_y
- `U_face, V_face` : face-normal velocities for H advection
"""
function compute_barotropic_rhs_nonCoriolis!(
    dM, dN, dH,
    M, N, H,
    state::State,
    grid::Grid,
    p::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int};
    H_in_u,
    H_in_v,
    termx,
    termy,
    U_face,
    V_face,
)
    Nx, Ny = p.Nx, p.Ny

    # Grid metrics
    dx_n2n_h  = grid.dx_n2n_h
    dy_n2n_h  = grid.dy_n2n_h
    dx_face_h = grid.dx_face_h
    dy_face_h = grid.dy_face_h
    dArea_h   = grid.dArea_h
    lat_u     = grid.lat_u
    lat_v     = grid.lat_v

    # Scalars
    g      = FT(p.g)
    ρ1     = FT(p.rho1)
    ρ2     = FT(p.rho2)
    hmin   = FT(p.hmin)
    Rearth = FT(p.earthRadius)

    # Forcing aliases
    forc    = state.forc
    taux_sf = forc.taux_sf
    tauy_sf = forc.tauy_sf
    taux_bt = forc.taux_bt
    tauy_bt = forc.tauy_bt

    # ----------------------------------------
    # 1. Wall BC for N (no-normal flow at N/S)
    # ----------------------------------------
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(N, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(H, Nx, Ny)

    # ----------------------------------------
    # 2. Reconstruct H at u/v points
    # ----------------------------------------
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(H_in_u, H, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(H_in_v, H, Nx, Ny, hmin)

    # ----------------------------------------
    # 3. Pressure-gradient contributions
    #    termx, termy = ∂H/∂x, ∂H/∂y at faces
    # ----------------------------------------
    @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
        termx, termy,
        H,
        dx_n2n_h, dy_n2n_h,
        Nx, Ny
    )

    # Convert to pressure-gradient forces at u/v points:
    # pres_grad_x ≈ -g * H_in_u * ∂H/∂x
    # pres_grad_y ≈ -g * H_in_v * ∂H/∂y
    @. termx = termx * (-g * H_in_u)
    @. termy = termy * (-g * H_in_v)

    # Initialize dM, dN with pressure gradient contributions
    @. dM = termx
    @. dN = termy

    # ----------------------------------------
    # 4. Curvature terms (metric terms)
    #    Overwrite termx, termy with curvature contributions,
    #    then add them into dM, dN.
    # ----------------------------------------
    @cuda threads=threads2 blocks=blocks2 k_calc_curvature_terms!(
        termx, termy,
        M, N,
        H_in_u, H_in_v,
        lat_u, lat_v,
        Nx, Ny,
        Rearth
    )

    @. dM = dM + termx
    @. dN = dN + termy

    # ----------------------------------------
    # 5. Forcing (wind + bottom stress)
    #    termx, termy get forcing; add into dM, dN.
    # ----------------------------------------
    @. termx = taux_sf / ρ1 + taux_bt / ρ2
    @. termy = tauy_sf / ρ1 + tauy_bt / ρ2

    @. dM = dM + termx
    @. dN = dN + termy

    # ----------------------------------------
    # 6. Mass equation: dH = -∇·(U H)
    # ----------------------------------------

    # Guard division via H_in_u/H_in_v already using hmin
    @. U_face = M / H_in_u
    @. V_face = N / H_in_v
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(V_face, Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        dH,         # output: (-∇·(U H))
        H,
        U_face, V_face,
        dx_face_h, dy_face_h,
        dArea_h,
        Nx, Ny
    )

    # dH already holds (-∇·(U H))
    return nothing
end


# ============================================================
# Single RK4 barotropic step (non-Coriolis + semi-implicit Coriolis)
# ============================================================

"""
    rk4_step_barotropic_once!(
        state::State,
        grid::Grid,
        p::Params,
        dt::FT;
        threads1, blocks1, threads2, blocks2
    )

Perform one RK4 time step of size `dt` for the barotropic variables
(M, N, H), treating:

- pressure, curvature, forcing, and mass divergence with **explicit RK4**
- Coriolis via a **single semi-implicit rotation** (k_add_coriolisforce!)
- Shapiro smoothing on H

Mutates `state.prog.M`, `state.prog.N`, `state.prog.H` in place.
"""
function rk4_step_barotropic_once!(
    state::State,
    grid::Grid,
    p::Params,
    dt::FT;
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
)
    Nx, Ny = p.Nx, p.Ny

    prog = state.prog

    M = prog.M
    N = prog.N
    H = prog.H

    # Scalars
    Ω       = FT(p.Ω)
    smoothϵ = FT(p.smoothing_eps)

    # Grid lats for Coriolis
    lat_u = grid.lat_u
    lat_v = grid.lat_v

    # RK4 weights
    dt_sixth = dt / FT(6)
    dt_third = dt / FT(3)
    dt_half  = dt / FT(2)

    temp = state.temp

    # --------------------------------------------------------
    # Alias Temporary arrays for RK4 + scratch (no allocations)
    # --------------------------------------------------------

    # Accumulated result q_accum = q_n + sum( weights * k )
    M_accum = temp.temp_var_x1
    N_accum = temp.temp_var_y1
    H_accum = temp.temp_var_x2

    # Stage state q_stage (M_stage, N_stage, H_stage)
    M_stage = temp.temp_var_y2
    N_stage = temp.temp_var_x3
    H_stage = temp.temp_var_y3

    # Current stage tendency k_q
    k_M = temp.temp_var_x4
    k_N = temp.temp_var_y4
    k_H = temp.temp_var_x5

    # Scratch for RHS
    H_in_u = temp.temp_var_y5
    H_in_v = temp.temp_var_x6
    termx  = temp.temp_var_y6
    termy  = temp.temp_var_x7
    U_face = temp.temp_var_y7
    V_face = temp.temp_var_x8

    # For Coriolis and Shapiro
    M_old  = temp.temp_var_y8
    N_old  = temp.temp_var_x9
    H_star = temp.temp_var_y9

    # -------------------------
    # Initialize accumulators
    # -------------------------
    @. M_accum = M
    @. N_accum = N
    @. H_accum = H

    # ========================================================
    # Stage 1: k1 = f(M, N, H)
    # ========================================================
    @. M_stage = M
    @. N_stage = N
    @. H_stage = H

    compute_barotropic_rhs_nonCoriolis!(
        k_M, k_N, k_H,
        M_stage, N_stage, H_stage,
        state, grid, p,
        threads1, blocks1, threads2, blocks2;
        H_in_u=H_in_u, H_in_v=H_in_v,
        termx=termx, termy=termy,
        U_face=U_face, V_face=V_face,
    )

    # Accumulate dt/6 * k1
    @. M_accum = M_accum + dt_sixth * k_M
    @. N_accum = N_accum + dt_sixth * k_N
    @. H_accum = H_accum + dt_sixth * k_H

    # Stage state for k2: q_stage = q_n + dt/2 * k1
    @. M_stage = M + dt_half * k_M
    @. N_stage = N + dt_half * k_N
    @. H_stage = H + dt_half * k_H

    # ========================================================
    # Stage 2: k2 = f(M_stage, N_stage, H_stage)
    # ========================================================
    compute_barotropic_rhs_nonCoriolis!(
        k_M, k_N, k_H,
        M_stage, N_stage, H_stage,
        state, grid, p,
        threads1, blocks1, threads2, blocks2;
        H_in_u=H_in_u, H_in_v=H_in_v,
        termx=termx, termy=termy,
        U_face=U_face, V_face=V_face,
    )

    # Accumulate dt/3 * k2
    @. M_accum = M_accum + dt_third * k_M
    @. N_accum = N_accum + dt_third * k_N
    @. H_accum = H_accum + dt_third * k_H

    # Stage state for k3: q_stage = q_n + dt/2 * k2
    @. M_stage = M + dt_half * k_M
    @. N_stage = N + dt_half * k_N
    @. H_stage = H + dt_half * k_H

    # ========================================================
    # Stage 3: k3 = f(M_stage, N_stage, H_stage)
    # ========================================================
    compute_barotropic_rhs_nonCoriolis!(
        k_M, k_N, k_H,
        M_stage, N_stage, H_stage,
        state, grid, p,
        threads1, blocks1, threads2, blocks2;
        H_in_u=H_in_u, H_in_v=H_in_v,
        termx=termx, termy=termy,
        U_face=U_face, V_face=V_face,
    )

    # Accumulate dt/3 * k3
    @. M_accum = M_accum + dt_third * k_M
    @. N_accum = N_accum + dt_third * k_N
    @. H_accum = H_accum + dt_third * k_H

    # Stage state for k4: q_stage = q_n + dt * k3
    @. M_stage = M + dt * k_M
    @. N_stage = N + dt * k_N
    @. H_stage = H + dt * k_H

    # ========================================================
    # Stage 4: k4 = f(M_stage, N_stage, H_stage)
    # ========================================================
    compute_barotropic_rhs_nonCoriolis!(
        k_M, k_N, k_H,
        M_stage, N_stage, H_stage,
        state, grid, p,
        threads1, blocks1, threads2, blocks2;
        H_in_u=H_in_u, H_in_v=H_in_v,
        termx=termx, termy=termy,
        U_face=U_face, V_face=V_face,
    )

    # Accumulate dt/6 * k4
    @. M_accum = M_accum + dt_sixth * k_M
    @. N_accum = N_accum + dt_sixth * k_N
    @. H_accum = H_accum + dt_sixth * k_H

    # -------------------------
    # Write back explicit RK4 result
    # -------------------------
    M .= M_accum
    N .= N_accum
    H .= H_accum

    # -------------------------
    # Semi-implicit Coriolis rotation
    # -------------------------
    M_old .= M
    N_old .= N

    # Use stage arrays as M_star, N_star
    M_stage .= M
    N_stage .= N

    @cuda threads=threads2 blocks=blocks2 k_add_coriolisforce!(
        M, N,
        M_old, N_old,
        M_stage, N_stage,
        lat_u, lat_v,
        Nx, Ny,
        dt, Ω
    )

    # -------------------------
    # Shapiro filter on H
    # -------------------------
    H_star .= H

    @cuda threads=threads2 blocks=blocks2 k_apply_shapiro_filter!(
        H,
        H_star,
        smoothϵ,
        Nx, Ny
    )

    return nothing
end


# ============================================================
# Public barotropic driver with subcycling
# ============================================================

"""
    step_barotropic!(
        state::State,
        grid::Grid,
        p::Params;
        threads1, blocks1, threads2, blocks2
    )

Advance the barotropic variables (M, N, H) by performing
`p.M` RK4 substeps of size `p.dtBT`.

Mutates:
- `state.prog.M`
- `state.prog.N`
- `state.prog.H`

Also sets:
- `state.prog.H_old` = H at the start of the baroclinic step.
"""
function step_barotropic!(
    state::State,
    grid::Grid,
    p::Params;
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
)
    prog = state.prog

    # Save H at start of the *baroclinic* step for coupling
    prog.H_old .= prog.H

    dt_bt  = FT(p.dtBT)
    nsteps = Int(p.M)

    for _ = 1:nsteps
        rk4_step_barotropic_once!(
            state, grid, p, dt_bt;
            threads1=threads1,
            blocks1=blocks1,
            threads2=threads2,
            blocks2=blocks2,
        )
    end

    return nothing
end
