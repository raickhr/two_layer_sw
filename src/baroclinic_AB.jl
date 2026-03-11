############################################################
# baroclinic.jl
#
# Baroclinic (internal-mode) time stepping for the two-layer
# rotating shallow-water model on a C-grid.
#
# This module advances the slow internal modes:
#   - Layer 1: (m1, n1, h1)
#   - Layer 2: (m2, n2, h2)
# over one baroclinic time step p.dt.
#
# Time stepping in this implementation:
#
#   - 3rd-order Adams–Bashforth (AB3) for non-Coriolis
#     baroclinic momentum tendencies:
#       * curvature (metric) terms
#       * surface & bottom stress splitting (via H_old)
#       * biharmonic viscosity (∇⁴ on u, v)
#       * WENO-Z flux-form advection of m, n
#
#   - WENO-Z flux-form advection for each layer thickness h:
#       h_t + ∇·(u h) = 0
#     in a predictor–corrector fashion (h* then hⁿ⁺¹).
#
#   - Baroclinic pressure / form drag based on ∇h₂ and, depending
#     on `mode_split`, barotropic PGF based on ∇H, are applied
#     in the corrector step.
#
#   - Semi-implicit Coriolis rotation via k_add_coriolisforce!,
#     applied in the corrector to (m_star, n_star) after the
#     pressure-gradient terms have been added.
#
#   - Weak Shapiro filter on h (10× weaker than barotropic
#     smoothing) at the end of the corrector.
#
# Scratch space:
#   - Uses state.temp::Temporary (temp_var_x1..x6, temp_var_y1..y6)
#   - Uses state.intp::Interpolated for thickness reconstructions
#   - Uses state.intm::Intermediate for star fields (m*, n*, h*, H*)
#
# History:
#   - Baroclinic AB3 histories live in state.hist:
#       r1_tm{0,1,2}_in_u, r1_tm{0,1,2}_in_v  (layer 1)
#       r2_tm{0,1,2}_in_u, r2_tm{0,1,2}_in_v  (layer 2)
############################################################


# ============================================================
# PREDICTOR: baroclinic AB3 + thickness predictor
# ============================================================

"""
    predictor_baroclinic!(m_star, n_star, h_star, r_tm0_u, r_tm1_u, r_tm2_u,
                          r_tm0_v, r_tm1_v, r_tm2_v,
                          m, n, h, H_old,
                          temp, intp, forc, grid, p,
                          threads1, blocks1, threads2, blocks2;
                          isFirstTimeStep=false, layer=1, mode_split=true)

Baroclinic predictor step for a single layer:

  • Updates `(m_star, n_star)` using AB3 on the non-Coriolis tendencies:
      - biharmonic viscosity
      - surface/bottom stress (optionally barotropically split)
      - metric/curvature terms
      - WENO-Z advection of m and n

  • Updates `h_star` with a forward-Euler step using WENO-Z flux-form
    advection of h:
        h_t + ∇·(u h) = 0,
    here using velocities derived from the **current** layer transports
    (m, n) and face-thicknesses (h_in_u, h_in_v).

Arguments:
  - `layer` == 1 or 2 selects which layer thickness (h1/h2) is active.
  - `mode_split` controls barotropic/baroclinic stress splitting.
"""
function predictor_baroclinic!(
    m_star, n_star, h_star,
    r_tm0_u, r_tm1_u, r_tm2_u,
    r_tm0_v, r_tm1_v, r_tm2_v,
    m, n, h, H_old,
    temp::Temporary,
    intp::Interpolated,
    forc::Forcing,
    grid::Grid,
    p::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int};
    isFirstTimeStep::Bool=false,
    layer::Int=1,
    mode_split::Bool=true,
)
    # Grid size and metrics
    Nx        = Int(p.Nx)
    Ny        = Int(p.Ny)
    lat_u     = grid.lat_u
    lat_v     = grid.lat_v
    dx_face_h = grid.dx_face_h
    dy_face_h = grid.dy_face_h
    dArea_h   = grid.dArea_h
    dx_face_u = grid.dx_face_u
    dy_face_u = grid.dy_face_u
    dArea_u   = grid.dArea_u
    dx_face_v = grid.dx_face_v
    dy_face_v = grid.dy_face_v
    dArea_v   = grid.dArea_v

    # Interpolated thickness (from State.intp)
    H_in_u = intp.H_in_u
    H_in_v = intp.H_in_v

    h1_in_u = intp.h1_in_u
    h1_in_v = intp.h1_in_v
    h2_in_u = intp.h2_in_u
    h2_in_v = intp.h2_in_v

    # Active-layer thickness at faces
    h_in_u = (layer == 1) ? h1_in_u : h2_in_u
    h_in_v = (layer == 1) ? h1_in_v : h2_in_v

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
    dt     = FT(p.dt)
    Ω      = FT(p.Ω)

    # Adams–Bashforth 3 weights
    Wn   = FT(23) / FT(12)
    Wnm1 = FT(-16) / FT(12)
    Wnm2 = FT(5)  / FT(12)

    # ----------------------------------------
    # PREDICTOR: Momentum conservation (AB3)
    # ----------------------------------------

    # 1. Apply wall BC for n (no-normal flow at N/S) and enforce walls on h, H_old
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(H_old, Nx, Ny)

    # 2. Reconstruct h (active layer) and H_old at u/v points
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h_in_u, h,     Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h_in_v, h,     Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(H_in_u, H_old, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(H_in_v, H_old, Nx, Ny, hmin)

    # 3. Biharmonic viscosity term, using u = m/h_in_u, v = n/h_in_v
    biharmonic_m = temp.temp_var_x1
    biharmonic_n = temp.temp_var_y1

    u = temp.temp_var_x2
    v = temp.temp_var_y2

    @. u = m / h_in_u
    @. v = n / h_in_v

    buf_x = temp.temp_var_x3
    buf_y = temp.temp_var_y3

    calculate_biharmonic_term!(
        biharmonic_m, biharmonic_n,
        u, v,
        h_in_u, h_in_v,
        buf_x, buf_y,
        grid, threads2, blocks2,
        ν, Nx, Ny,
    )

    @. r_tm0_u = biharmonic_m
    @. r_tm0_v = biharmonic_n

    # 4. Forcing (wind + bottom stress) with optional barotropic split
    forc_m = temp.temp_var_x1
    forc_n = temp.temp_var_y1

    if mode_split
        if layer == 1
            @. forc_m = taux_sf/ρ1 - h_in_u/H_in_u * (taux_sf/ρ1 + taux_bt/ρ2)
            @. forc_n = tauy_sf/ρ1 - h_in_v/H_in_v * (tauy_sf/ρ1 + tauy_bt/ρ2)
        else # layer == 2
            @. forc_m = taux_bt/ρ2 - h_in_u/H_in_u * (taux_sf/ρ1 + taux_bt/ρ2)
            @. forc_n = tauy_bt/ρ2 - h_in_v/H_in_v * (tauy_sf/ρ1 + tauy_bt/ρ2)
        end
    else
        if layer == 1
            @. forc_m = taux_sf/ρ1
            @. forc_n = tauy_sf/ρ1
        else
            @. forc_m = taux_bt/ρ2
            @. forc_n = tauy_bt/ρ2
        end
    end

    @. r_tm0_u = r_tm0_u + forc_m
    @. r_tm0_v = r_tm0_v + forc_n

    # 5. Curvature (metric) terms
    curv_x = temp.temp_var_x1
    curv_y = temp.temp_var_y1

    @cuda threads=threads2 blocks=blocks2 k_calc_curvature_terms!(
        curv_x, curv_y,
        m, n,
        h_in_u, h_in_v,
        lat_u, lat_v,
        Nx, Ny,
        Rearth,
    )

    @. r_tm0_u = r_tm0_u + curv_x
    @. r_tm0_v = r_tm0_v + curv_y

    ######################################
    # 6. WENO-Z advection of m and n
    ######################################
    adv_m = temp.temp_var_x1
    adv_n = temp.temp_var_y1

    # u-cells (advecting m)
    u_face_for_u = temp.temp_var_x2
    v_face_for_u = temp.temp_var_y2

    @cuda threads=threads2 blocks=blocks2 k_calc_faceVels_for_ucell!(
        u_face_for_u, v_face_for_u,
        m, n,
        h,
        Nx, Ny,
        hmin,
    )
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(v_face_for_u, Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        adv_m,
        m,
        u_face_for_u, v_face_for_u,
        dx_face_u, dy_face_u,
        dArea_u,
        Nx, Ny,
        UGRID,
    )
    @. r_tm0_u = r_tm0_u + adv_m

    # v-cells (advecting n)
    u_face_for_v = temp.temp_var_x2
    v_face_for_v = temp.temp_var_y2

    @cuda threads=threads2 blocks=blocks2 k_calc_faceVels_for_vcell!(
        u_face_for_v, v_face_for_v,
        m, n,
        h,
        Nx, Ny,
        hmin,
    )
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(v_face_for_v, Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        adv_n,
        n,
        u_face_for_v, v_face_for_v,
        dx_face_v, dy_face_v,
        dArea_v,
        Nx, Ny,
        VGRID,
    )
    @. r_tm0_v = r_tm0_v + adv_n

    # 7. Initialize History at first step
    if isFirstTimeStep
        @. r_tm1_u = r_tm0_u
        @. r_tm2_u = r_tm0_u

        @. r_tm1_v = r_tm0_v
        @. r_tm2_v = r_tm0_v
    end

    # 8. AB3 predictor for m_star, n_star
    #@. m_star = m + dt * (Wn * r_tm0_u + Wnm1 * r_tm1_u + Wnm2 * r_tm2_u)
    #@. n_star = n + dt * (Wn * r_tm0_v + Wnm1 * r_tm1_v + Wnm2 * r_tm2_v)

    @. m_star = m + dt * r_tm0_u 
    @. n_star = n + dt * r_tm0_v 

    # Wall BC on predicted n_star
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n_star, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(m_star, Nx, Ny)

    # 9. Rotate History: r_tm2 <= r_tm1, r_tm1 <= r_tm0
    r_tm2_u .= r_tm1_u
    r_tm1_u .= r_tm0_u

    r_tm2_v .= r_tm1_v
    r_tm1_v .= r_tm0_v


    # ----------------------------------------
    # PREDICTOR: Mass conservation (h_star)
    # ----------------------------------------

    # Uses velocities derived from **current** layer transports (m, n)
    # and face-thicknesses (h_in_u, h_in_v) to advance h with FE.
    @. u = m / h_in_u
    @. v = n / h_in_v

    minus_div_uh = temp.temp_var_x3

    # Assumption: k_calc_WENOZ_flux2d! returns - ∇·(u h),
    # so h_star = h + dt * minus_div_uh is a forward Euler step.
    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        minus_div_uh,
        h,
        u, v,
        dx_face_h, dy_face_h,
        dArea_h,
        Nx, Ny,
        HGRID,
    )

    h_star .= h .+ dt .* minus_div_uh
    
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h_star, Nx, Ny)

    return nothing
end


# ============================================================
# CORRECTOR: pressure gradient + Coriolis + h-corrector
# ============================================================

"""
    corrector_baroclinic!(m, n, h,
                          h1_old, h2_old, H_old,
                          m_star, n_star,
                          h1_star, h2_star, H_star,
                          temp, intp, grid, p,
                          threads1, blocks1, threads2, blocks2;
                          layer, mode_split=true)

Baroclinic corrector step for a single layer:

  • Builds time-centered thicknesses (h₁, h₂, H) and computes
    baroclinic pressure-gradient terms (form drag) from ∇h₂.

  • When `mode_split == false`, also computes barotropic PGF based on
    ∇H_center and adds it to the layer PGF.

  • Adds these PGF terms to (m_star, n_star), then applies a
    semi-implicit Coriolis update via `k_add_coriolisforce!` to obtain
    (m_plus, n_plus).

  • Updates the layer thickness h using a second WENO-Z flux-form step
    with time-centered velocities, and applies a weak Shapiro filter.
"""
function corrector_baroclinic!(
    m, n, h, 
    h1_old, h2_old, H_old,
    m_star, n_star, h1_star, h2_star, H_star,
    temp::Temporary,
    intp::Interpolated,
    grid::Grid,
    p::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int};
    layer::Int,
    mode_split::Bool=true,
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
    dt      = FT(p.dt)
    g       = FT(p.g)
    gp      = FT(p.gp)
    Ω       = FT(p.Ω)
    smoothϵ = FT(0.1) * FT(p.smoothing_eps)

    # Interpolated layer thicknesses for PGF coefficients
    h1c_in_u = intp.h1_in_u
    h1c_in_v = intp.h1_in_v
    h2c_in_u = intp.h2_in_u
    h2c_in_v = intp.h2_in_v

    H_center  = temp.temp_var_x5
    h1_center = temp.temp_var_x6
    h2_center = temp.temp_var_y6

    # Build total H_star = h1_star + h2_star for optional barotropic PGF
    @. H_star = h1_star + h2_star

    # Time-centered thicknesses
    @. H_center  = FT(0.5) * (H_old  + H_star)
    @. h1_center = FT(0.5) * (h1_old + h1_star)
    @. h2_center = FT(0.5) * (h2_old + h2_star)

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(H_center, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h1_center, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h2_center, Nx, Ny)

    # Active-layer face thickness and center thickness aliases
    if layer == 1
        hc_in_u  = h1c_in_u
        hc_in_v  = h1c_in_v
        h_center = h1_center
    else
        hc_in_u  = h2c_in_u
        hc_in_v  = h2c_in_v
        h_center = h2_center
    end

    # ----------------------------------------
    # CORRECTOR: Pressure gradient
    # ----------------------------------------

    press_gradx = temp.temp_var_x1
    press_grady = temp.temp_var_y1

    h2_gradx = temp.temp_var_x2
    h2_grady = temp.temp_var_y2

    # Baroclinic part: ∇h2_center (proxy for interface displacement ∇ξ)
    @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
        h2_gradx, h2_grady,
        h2_center,
        dx_n2n_h, dy_n2n_h,
        Nx, Ny,
    )

    # Reconstruct h1_center, h2_center at faces for PGF coefficients
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h1c_in_u, h1_center, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h1c_in_v, h1_center, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h2c_in_u, h2_center, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h2c_in_v, h2_center, Nx, Ny, hmin)

    if mode_split
        if layer == 1
            # Layer 1: + g' h1 ∇h2 (pure baroclinic PGF in split mode)
            @. press_gradx = gp * h1c_in_u * h2_gradx 
            @. press_grady = gp * h1c_in_v * h2_grady
        else
            # Layer 2: - g' h2 ∇h2 (pure baroclinic PGF in split mode)
            @. press_gradx = -gp * h2c_in_u * h2_gradx 
            @. press_grady = -gp * h2c_in_v * h2_grady
        end

    else
        # No mode splitting: build barotropic + baroclinic PGF from
        # total thickness H_center and interface proxy h2_center.
        H_gradx = temp.temp_var_x3
        H_grady = temp.temp_var_y3

        @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
            H_gradx, H_grady,
            H_center,
            dx_n2n_h, dy_n2n_h,
            Nx, Ny,
        )

        if layer == 1
            # Layer 1: purely barotropic PGF
            @. press_gradx = -g * h1c_in_u * H_gradx 
            @. press_grady = -g * h1c_in_v * H_grady
        else
            # Layer 2: barotropic + baroclinic PGF
            @. press_gradx = -g * h2c_in_u * H_gradx - gp * h2c_in_u * h2_gradx 
            @. press_grady = -g * h2c_in_v * H_grady - gp * h2c_in_v * h2_grady
        end
    end

    # ----------------------------------------
    # CORRECTOR: Apply Coriolis and PGF (final m, n)
    # ----------------------------------------

    # Add PGF explicitly to (m_star, n_star) and then apply
    # semi-implicit Coriolis via k_add_coriolisforce!.
    @. m_star = m_star + dt * press_gradx
    @. n_star = n_star + dt * press_grady

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n_star, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(m_star, Nx, Ny)

    m_plus = temp.temp_var_x2
    n_plus = temp.temp_var_y2

    @cuda threads=threads2 blocks=blocks2 k_add_coriolisforce!(
        m_plus, n_plus, 
        m_star, n_star,
        m, n,
        lat_u, lat_v,
        Nx, Ny, Ω, dt,
    )

    # Enforce no-normal flow at N/S walls after Coriolis rotation
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n_plus, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(m_star, Nx, Ny)

    # ----------------------------------------
    # CORRECTOR: Mass conservation (final h)
    # ----------------------------------------

    u_center = temp.temp_var_x3
    v_center = temp.temp_var_y3

    # Time-centered velocities (mⁿ, mⁿ⁺¹) for advection of h
    @. u_center = FT(0.5) * (m + m_plus) / hc_in_u
    @. v_center = FT(0.5) * (n + n_plus) / hc_in_v

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(v_center, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_u!(u_center, Nx, Ny)

    minus_div_uh = temp.temp_var_x4
    h_plus       = temp.temp_var_y4

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        minus_div_uh,
        h_center,
        u_center, v_center,
        dx_face_h, dy_face_h,
        dArea_h,
        Nx, Ny,
        HGRID,
    )

    h_plus .= h .+ dt .* minus_div_uh
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h_plus, Nx, Ny)

    # -------------------------
    # Shapiro filter on h
    # -------------------------
    @cuda threads=threads2 blocks=blocks2 k_apply_shapiro_filter!(
        h,
        h_plus,
        smoothϵ,
        Nx, Ny,
    )

    @. m = m_plus
    @. n = n_plus

    return nothing
end


# ============================================================
# Public API: baroclinic step for both layers
# ============================================================

"""
    step_baroclinic!(state, grid, p;
                     threads1, blocks1, threads2, blocks2,
                     step, mode_split=true)

Advance **both baroclinic layers** (1 then 2) by one baroclinic time
step `p.dt` using:

  • an AB3 + WENO-Z predictor for (m, n, h) in each layer, and
  • a corrector with baroclinic PGF (and, when `mode_split == false`,
    additional barotropic PGF), semi-implicit Coriolis via
    `k_add_coriolisforce!`, and an h-corrector with a Shapiro filter.

Arguments
---------
- `step`       : current baroclinic step index (1-based); used only
                 to decide when to initialize AB3 histories
                 (`isFirstTimeStep = (step == 1)`).
- `mode_split` : if true, use barotropic/baroclinic stress splitting
                 and omit barotropic PGF in the corrector; if false,
                 add a barotropic PGF term based on total thickness H
                 (and a combined barotropic + baroclinic PGF in layer 2).
"""
function step_baroclinic!(
    state::State,
    grid::Grid,
    p::Params;
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
    step::Int,
    mode_split::Bool = true,
)
    # Aliases
    prog = state.prog
    intp = state.intp
    intm = state.intm
    hist = state.hist
    temp = state.temp
    forc = state.forc

    # Layer 1 prognostics
    m1 = prog1.m
    n1 = prog1.n
    h1 = prog1.h

    # Layer 2 prognostics
    m2 = prog2.m
    n2 = prog2.n
    h2 = prog2.h

    # History variables for AB3
    H_old   = hist.H_old
    h1_old  = hist.h1_old
    h2_old  = hist.h2_old

    r1_tm0_u = hist.r1_tm0_in_u
    r1_tm1_u = hist.r1_tm1_in_u
    r1_tm2_u = hist.r1_tm2_in_u

    r1_tm0_v = hist.r1_tm0_in_v
    r1_tm1_v = hist.r1_tm1_in_v
    r1_tm2_v = hist.r1_tm2_in_v

    r2_tm0_u = hist.r2_tm0_in_u
    r2_tm1_u = hist.r2_tm1_in_u
    r2_tm2_u = hist.r2_tm2_in_u

    r2_tm0_v = hist.r2_tm0_in_v
    r2_tm1_v = hist.r2_tm1_in_v
    r2_tm2_v = hist.r2_tm2_in_v

    # Intermediate variables
    m1_star = intm.m1_star
    n1_star = intm.n1_star
    h1_star = intm.h1_star

    m2_star = intm.m2_star
    n2_star = intm.n2_star
    h2_star = intm.h2_star

    H_star  = intm.H_star

    # Predictor: layer 1
    predictor_baroclinic!(
        m1_star,  n1_star,  h1_star,
        r1_tm0_u, r1_tm1_u, r1_tm2_u,
        r1_tm0_v, r1_tm1_v, r1_tm2_v,
        m1, n1, h1, H_old,
        temp, intp, forc, grid, p,
        threads1, blocks1,
        threads2, blocks2;
        isFirstTimeStep = (step == 1),
        layer           = 1,
        mode_split      = mode_split,
    )

    # Predictor: layer 2
    predictor_baroclinic!(
        m2_star,  n2_star,  h2_star,
        r2_tm0_u, r2_tm1_u, r2_tm2_u,
        r2_tm0_v, r2_tm1_v, r2_tm2_v,
        m2, n2, h2, H_old,
        temp, intp, forc, grid, p,
        threads1, blocks1,
        threads2, blocks2;
        isFirstTimeStep = (step == 1),
        layer           = 2,
        mode_split      = mode_split,
    )

    # Corrector: layer 1
    corrector_baroclinic!(
        m1, n1, h1, 
        h1_old, h2_old, H_old,
        m1_star, n1_star, h1_star, h2_star, H_star,
        temp, intp, grid, p,
        threads1, blocks1,
        threads2, blocks2;
        layer      = 1,
        mode_split = mode_split,
    )

    # Corrector: layer 2
    corrector_baroclinic!(
        m2, n2, h2, 
        h1_old, h2_old, H_old,
        m2_star, n2_star, h1_star, h2_star, H_star,
        temp, intp, grid, p,
        threads1, blocks1,
        threads2, blocks2;
        layer      = 2,
        mode_split = mode_split,
    )

    return nothing
end
