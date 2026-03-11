# Barotropic external-mode update (one substep)
#
#   Time levels:        n                    n+1
#                        |--------------------|
#
#   1) Predictor (explicit AB3 + WENO-Z)
#
#       M^n, N^n, H^n
#          |
#          |  AB3 on non-Coriolis RHS (visc + curvature + forcing)
#          v
#       M*, N*          (predicted transports)
#          |
#          |  WENO-Z on mass: H_t + ∇·(U H) = 0, U* = M*/H
#          v
#       H*              (predicted thickness)
#
#   2) Corrector (PG + semi-implicit Coriolis + centered mass)
#
#       M*, N*, H*, H^n
#          |
#          |  Time-centered thickness: H_c = (H^n + H*)/2
#          |  Compute ∇H_c and reconstruct H_c at u/v
#          v
#   (M*, N*) + PG       (add -g H_c ∇H_c at u/v to M*, N*)
#          |
#          |  Semi-implicit Coriolis rotation via k_apply_coriolis!
#          v
#       M^{n+1}, N^{n+1}
#          |
#          |  Center transports in time: M_c = (M^n + M^{n+1})/2
#          |  Use U_c = M_c / H_c in WENO-Z mass update
#          v
#       H^{n+1} (from H^n + dt *[-∇·(U_c H_c)] + Shapiro filter)
#
#   Final state of barotropic mode at end of substep:
#       (M^{n+1}, N^{n+1}, H^{n+1})

############################################################
# barotropic.jl
#
# Barotropic (external-mode) subcycling for the two-layer
# rotating shallow-water model.
#
# This module advances the fast external mode (M, N, H) by
# performing p.M barotropic substeps of size p.dtBT for
# each baroclinic timestep.
#
# Time stepping in this implementation:
#
#   - 3rd-order Adams–Bashforth (AB3) for non-Coriolis
#     barotropic momentum tendencies:
#       * biharmonic viscosity
#       * curvature (metric) terms
#       * surface wind stress + bottom stress
#
#   - WENO-Z flux-form advection for the mass equation
#     (H_t + ∇·(U H) = 0) in a predictor–corrector fashion.
#     Only H is advected; there is no explicit barotropic
#     advection of M, N here.
#
#   - Combined pressure-gradient + semi-implicit Coriolis
#     rotation via k_apply_coriolis! in the corrector step.
#
# NOTES:
#   - This is NOT the full MOM6-style implicit barotropic
#     solver; external gravity waves remain explicit and
#     are subject to a barotropic CFL restriction.
############################################################

function predictor_barotropic!(
    M_star, N_star, H_star,
    Rtm0_u, Rtm1_u, Rtm2_u,
    Rtm0_v, Rtm1_v, Rtm2_v,
    M, N, H,
    temp::Temporary,
    intp::Interpolated,
    forc::Forcing,
    grid::Grid,
    p::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
    isFirstTimeStep::Bool=false,
)
    # Grid size and metrics
    Nx        = Int(p.Nx)
    Ny        = Int(p.Ny)
    lat_u     = grid.lat_u
    lat_v     = grid.lat_v
    dx_face_h = grid.dx_face_h
    dy_face_h = grid.dy_face_h
    dArea_h   = grid.dArea_h

    # Interpolated thickness at u/v points (used and overwritten each call)
    H_in_u = intp.H_in_u
    H_in_v = intp.H_in_v

    # Forcing aliases
    taux_sf = forc.taux_sf
    tauy_sf = forc.tauy_sf
    taux_bt = forc.taux_bt
    tauy_bt = forc.tauy_bt

    # Scalars
    ρ1     = FT(p.rho1)
    ρ2     = FT(p.rho2)
    hmin   = FT(p.hmin)
    Rearth = FT(p.earthRadius)
    ν      = FT(p.nu)
    dt     = FT(p.dtBT)

    # Adams–Bashforth 3 weights
    Wn   = FT(23) / FT(12)
    Wnm1 = FT(-16) / FT(12)
    Wnm2 = FT(5)  / FT(12)

    # ----------------------------------------
    # PREDICTOR: Momentum conservation (AB3)
    # ----------------------------------------

    # 1. Apply wall BC for N, H (no-normal flow at N/S for N)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(N, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(H, Nx, Ny)

    # 2. Reconstruct H at u/v points from current H
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(H_in_u, H, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(H_in_v, H, Nx, Ny, hmin)

    # 3. Biharmonic viscosity term, using U = M/H_in_u, V = N/H_in_v
    biharmonic_M = temp.temp_var_x1
    biharmonic_N = temp.temp_var_y1

    U = temp.temp_var_x2
    V = temp.temp_var_y2

    @. U = M / H_in_u
    @. V = N / H_in_v

    buf_x = temp.temp_var_x3
    buf_y = temp.temp_var_y3

    calculate_biharmonic_term!(
        biharmonic_M, biharmonic_N,
        U, V,
        H_in_u, H_in_v,
        buf_x, buf_y,
        grid, threads2, blocks2,
        ν, Nx, Ny,
    )

    @. Rtm0_u = biharmonic_M
    @. Rtm0_v = biharmonic_N

    # 4. Forcing (wind + bottom stress), reusing temp_var_x1/y1
    forc_M = temp.temp_var_x1
    forc_N = temp.temp_var_y1

    @. forc_M = taux_sf / ρ1 + taux_bt / ρ2
    @. forc_N = tauy_sf / ρ1 + tauy_bt / ρ2

    @. Rtm0_u = Rtm0_u + forc_M
    @. Rtm0_v = Rtm0_v + forc_N

    # 5. Curvature (metric) terms, again reusing temp_var_x1/y1
    curv_x = temp.temp_var_x1
    curv_y = temp.temp_var_y1

    @cuda threads=threads2 blocks=blocks2 k_calc_curvature_terms!(
        curv_x, curv_y,
        M, N,
        H_in_u, H_in_v,
        lat_u, lat_v,
        Nx, Ny,
        Rearth,
    )

    @. Rtm0_u = Rtm0_u + curv_x
    @. Rtm0_v = Rtm0_v + curv_y

    # 6. Initialize AB3 History at first step
    if isFirstTimeStep
        @. Rtm1_u = Rtm0_u
        @. Rtm2_u = Rtm0_u

        @. Rtm1_v = Rtm0_v
        @. Rtm2_v = Rtm0_v
    end

    # 7. AB3 predictor for M_star, N_star
    @. M_star = M + dt * (Wn * Rtm0_u + Wnm1 * Rtm1_u + Wnm2 * Rtm2_u)
    @. N_star = N + dt * (Wn * Rtm0_v + Wnm1 * Rtm1_v + Wnm2 * Rtm2_v)

    # Wall BC on predicted N_star
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(N_star, Nx, Ny)

    # 8. Rotate History: Rtm2 <= Rtm1, Rtm1 <= Rtm0
    Rtm2_u .= Rtm1_u
    Rtm1_u .= Rtm0_u

    Rtm2_v .= Rtm1_v
    Rtm1_v .= Rtm0_v

    # ----------------------------------------
    # PREDICTOR: Mass conservation (H_star)
    # ----------------------------------------

    # U*, V* from M_star, N_star (using same H_in_u/v as above)
    @. U = M_star / H_in_u
    @. V = N_star / H_in_v

    minus_div_UH = temp.temp_var_x3

    # Assumed: k_calc_WENOZ_flux2d! returns -∇·(U H)
    # so H_star = H + dt * minus_div_UH is a forward Euler step.
    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        minus_div_UH,
        H,
        U, V,
        dx_face_h, dy_face_h,
        dArea_h,
        Nx, Ny,
    )

    H_star .= H .+ dt .* minus_div_UH
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(H_star, Nx, Ny)

    return nothing
end


function corrector_barotropic!(
    M, N, H,
    M_star, N_star, H_star,
    temp::Temporary,
    intp::Interpolated,
    grid::Grid,
    p::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
)
    # Grid size and metrics
    Nx        = Int(p.Nx)
    Ny        = Int(p.Ny)
    lat_u     = grid.lat_u
    lat_v     = grid.lat_v
    dx_face_h = grid.dx_face_h
    dy_face_h = grid.dy_face_h
    dArea_h   = grid.dArea_h
    dx_n2n_h  = grid.dx_n2n_h
    dy_n2n_h  = grid.dy_n2n_h

    # Scalars
    hmin    = FT(p.hmin)
    dt      = FT(p.dtBT)
    g       = FT(p.g)
    Ω       = FT(p.Ω)
    smoothϵ = FT(p.smoothing_eps)

    # Interpolated H at u/v points (will be overwritten here)
    Hc_in_u = intp.H_in_u
    Hc_in_v = intp.H_in_v

    # ----------------------------------------
    # CORRECTOR: Pressure gradient + Coriolis
    # ----------------------------------------

    # 1. Time-centered H at h-points: H_center = (H + H_star)/2
    press_gradx = temp.temp_var_x1
    press_grady = temp.temp_var_y1
    Hcenter     = temp.temp_var_x5

    @. Hcenter = FT(0.5) * (H_star + H)

    # Compute ∇H_center on the h-grid
    @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
        press_gradx, press_grady,
        Hcenter,
        dx_n2n_h, dy_n2n_h,
        Nx, Ny,
    )

    # 2. Reconstruct H_center at u/v points
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(Hc_in_u, Hcenter, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(Hc_in_v, Hcenter, Nx, Ny, hmin)

    # 3. Convert to pressure-gradient forces at u/v points:
    #    pres_grad_x ≈ -g * Hc_in_u * ∂H_center/∂x
    #    pres_grad_y ≈ -g * Hc_in_v * ∂H_center/∂y
    @. press_gradx = (-g * Hc_in_u) * press_gradx
    @. press_grady = (-g * Hc_in_v) * press_grady

    # 4. Apply combined pressure gradient + semi-implicit Coriolis
    #    via k_apply_coriolis!, starting from M_star, N_star.
    M_plus = temp.temp_var_x2
    N_plus = temp.temp_var_y2

    @cuda threads=threads2 blocks=blocks2 k_apply_coriolis!(
        M_plus, N_plus,            # output
        M_star, N_star,            # input (pre-PG/Coriolis)
        press_gradx, press_grady,  # input (PG at u/v)
        lat_u, lat_v,              # input (latitudes at u/v)
        Nx, Ny, Ω, dt
    )

    # Enforce no-normal flow at N/S walls after PG + Coriolis
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(N_plus, Nx, Ny)

    # ----------------------------------------
    # CORRECTOR: Mass conservation (final H)
    # ----------------------------------------

    # Centered velocities U_center, V_center from M_center, H_c at faces
    U_center = temp.temp_var_x3
    V_center = temp.temp_var_y3

    @. U_center = FT(0.5) * (M + M_plus) / Hc_in_u
    @. V_center = FT(0.5) * (N + N_plus) / Hc_in_v

    minus_div_UH = temp.temp_var_x4
    H_plus       = temp.temp_var_y4

    # Assume k_calc_WENOZ_flux2d! returns -∇·(U_center H_center).
    # This gives H^{n+1} = H^n + dt * [ -∇·(U_center H_center) ].
    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        minus_div_UH,
        Hcenter,
        U_center, V_center,
        dx_face_h, dy_face_h,
        dArea_h,
        Nx, Ny,
    )

    H_plus .= H .+ dt .* minus_div_UH
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(H_plus, Nx, Ny)

    # -------------------------
    # Shapiro filter on H_plus
    # -------------------------
    @cuda threads=threads2 blocks=blocks2 k_apply_shapiro_filter!(
        H,      # output: filtered thickness
        H_plus, # input
        smoothϵ,
        Nx, Ny,
    )

    # Update prognostic barotropic transports
    @. M = M_plus
    @. N = N_plus

    return nothing
end


function step_barotropic_once!(
    state::State,
    grid::Grid,
    p::Params;
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
    isFirstTimeStep::Bool=false,
)
    # Aliases
    prog = state.prog
    intp = state.intp
    intm = state.intm
    hist = state.hist
    temp = state.temp
    forc = state.forc

    # Prognostic barotropic variables (on device)
    M = prog.M   # barotropic hu at u-points
    N = prog.N   # barotropic hv at v-points
    H = prog.H   # barotropic thickness / free-surface-equivalent

    # History variables for AB3 (non-Coriolis barotropic tendencies)
    Rtm0_u = hist.R_tm0_in_u
    Rtm1_u = hist.R_tm1_in_u
    Rtm2_u = hist.R_tm2_in_u

    Rtm0_v = hist.R_tm0_in_v
    Rtm1_v = hist.R_tm1_in_v
    Rtm2_v = hist.R_tm2_in_v

    # Intermediate variables
    M_star = intm.M_star
    N_star = intm.N_star
    H_star = intm.H_star

    # Predictor: AB3 + WENO-Z predictor for (M_star, N_star, H_star)
    predictor_barotropic!(
        M_star, N_star, H_star,
        Rtm0_u, Rtm1_u, Rtm2_u,
        Rtm0_v, Rtm1_v, Rtm2_v,
        M, N, H,
        temp, intp, forc, grid, p,
        threads1, blocks1,
        threads2, blocks2,
        isFirstTimeStep,
    )

    # Corrector: pressure gradient + Coriolis + centered mass update
    corrector_barotropic!(
        M, N, H,
        M_star, N_star, H_star,
        temp, intp, grid, p,
        threads1, blocks1,
        threads2, blocks2,
    )

    return nothing
end


"""
    step_barotropic!(
        state::State,
        grid::Grid,
        p::Params;
        threads1, blocks1, threads2, blocks2, step
    )

Advance the barotropic (external) mode variables `M`, `N`, and `H`
over one baroclinic time step by performing `p.M` barotropic
substeps of size `p.dtBT`.

Arguments
---------
- `state`   : model state containing prognostic, History, interp, temp,
              and forcing fields.
- `grid`    : grid/metric information.
- `p`       : parameter struct (contains Nx, Ny, dtBT, M, etc.).
- `threads1`, `blocks1` : CUDA launch config for 1D kernels.
- `threads2`, `blocks2` : CUDA launch config for 2D kernels.
- `step`    : baroclinic step index (1-based).

Notes
-----
- The AB3 History (`R_tm*`) is initialized only on the very first
  barotropic substep of the entire run: `step == 1 && substep == 1`.
- On later steps, the AB3 History is carried forward across calls.
"""
function step_barotropic!(
    state::State,
    grid::Grid,
    p::Params;
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
    step::Int,
)
    nsubsteps = Int(p.M)

    for substep = 1:nsubsteps
        step_barotropic_once!(
            state, grid, p;
            threads1 = threads1,
            blocks1  = blocks1,
            threads2 = threads2,
            blocks2  = blocks2,
            isFirstTimeStep = (step == 1 && substep == 1),
        )
    end

    return nothing
end
