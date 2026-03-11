############################################################
# baroclinic.jl
#
# Baroclinic (internal-mode) time stepping for a two-layer
# rotating shallow-water model on a C-grid.
#
# This module advances the slow internal modes:
#   - Layer 1: (m1, n1, h1)
#   - Layer 2: (m2, n2, h2)
#
# using a ROMS-style LF–AM3 scheme over one baroclinic time
# step p.dt:
#
#   1. Predictor (Leapfrog):
#        q^{n+1,*} = q^{n-1} + 2 Δt R(q^n)
#
#   2. Intermediate averaging (AM3):
#        q^{n+1/2} = (5/12) q^{n+1,*}
#                   + (2/3)  q^n
#                   - (1/12) q^{n-1}
#
#   3. Corrector (Euler with centered tendencies):
#        q^{n+1} = q^n + Δt R(q^{n+1/2})
#
# where q stands for (m, n, h) for each layer.
#
# prog::Prognostic holds the state at time n (and updated to n+1).
# hist::history holds 3-level histories:
#     *_tm0 = value at n−1
#     *_tm1 = value at n
# intm::intermediate holds the AM3 time-centered fields:
#     h1_star, h2_star, H_star
#     m1_star, m2_star, M_star
#     n1_star, n2_star, N_star
############################################################


# ============================================================
# Helper: initialize & rotate 3-level history
# ============================================================

"""
    initialize_baroclinic_history!(hist, prog)

Initialize the baroclinic 3-level history buffers for the LF–AM3
scheme.

After this call:

  * `*_tm0` = state at n−1 = current state (degenerate)
  * `*_tm1` = state at n   = current state

This is used at the very first baroclinic step (`step == 1`).
"""
function initialize_baroclinic_history!(hist::history, prog::Prognostic)
    # Layer-1 thickness / momentum
    @. hist.h1_tm0 = prog.h1
    @. hist.h1_tm1 = prog.h1

    @. hist.m1_tm0 = prog.m1
    @. hist.m1_tm1 = prog.m1

    @. hist.n1_tm0 = prog.n1
    @. hist.n1_tm1 = prog.n1

    # Layer-2 thickness / momentum
    @. hist.h2_tm0 = prog.h2
    @. hist.h2_tm1 = prog.h2

    @. hist.m2_tm0 = prog.m2
    @. hist.m2_tm1 = prog.m2

    @. hist.n2_tm0 = prog.n2
    @. hist.n2_tm1 = prog.n2

    return nothing
end


"""
    rotate_baroclinic_history!(hist, prog)

Rotate baroclinic history buffers after completing a full LF–AM3
step.

On entry, `prog` holds the *new* state at n+1. On exit:

  * `*_tm0` ← old `*_tm1`  (n−1 ← n)
  * `*_tm1` ← `prog.*`     (n   ← n+1)

so that at the next time step, the 3-level history corresponds to
(n−1, n, n+1).
"""
function rotate_baroclinic_history!(hist::history, prog::Prognostic)
    # Layer 1
    @. hist.h1_tm0 = hist.h1_tm1
    @. hist.h1_tm1 = prog.h1

    @. hist.m1_tm0 = hist.m1_tm1
    @. hist.m1_tm1 = prog.m1

    @. hist.n1_tm0 = hist.n1_tm1
    @. hist.n1_tm1 = prog.n1

    # Layer 2
    @. hist.h2_tm0 = hist.h2_tm1
    @. hist.h2_tm1 = prog.h2

    @. hist.m2_tm0 = hist.m2_tm1
    @. hist.m2_tm1 = prog.m2

    @. hist.n2_tm0 = hist.n2_tm1
    @. hist.n2_tm1 = prog.n2

    return nothing
end


# ============================================================
# PREDICTOR: LF + AM3 time-centered star fields
# ============================================================

"""
    predictor_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, p,
        threads1, blocks1,
        threads2, blocks2;
        isFirstTimeStep = false,
        layer           = 1,
        mode_split      = true,
    )

Baroclinic **predictor** for a single layer (1 or 2), implementing
the ROMS-style LF–AM3 scheme up to the AM3 time-centered “star”
state.

Mathematically, for each prognostic variable q ∈ {m, n, h}:

  1. Compute R(qⁿ) from the current fields in `prog`.
  2. Leapfrog predictor:
         q^{n+1,*} = q^{n-1} + 2 Δt R(qⁿ)
     where q^{n-1} is taken from the history buffers
     `hist.*_tm0`.  When `isFirstTimeStep == true`, we use
     q^{n-1} ≡ qⁿ so that the scheme degrades gracefully at
     start-up.
  3. AM3 time-centering:
         q^{n+1/2} = (5/12) q^{n+1,*}
                     + (2/3) qⁿ
                     - (1/12) q^{n-1}

The resulting time-centered fields `(h_star, m_star, n_star)` are
stored in `intm` (within h1_star/h2_star etc.) and interpreted as
q^{n+1/2} for use in the corrector.
"""
function predictor_baroclinic!(
    prog::Prognostic,
    hist::history,
    intm::intermediate,
    temp::Temporary,
    intp::interpolated,
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
    # --------------------------------------------------------
    # Grid & scalar aliases
    # --------------------------------------------------------
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
    dx_n2n_h  = grid.dx_n2n_h
    dy_n2n_h  = grid.dy_n2n_h
    f_u       = grid.f_u
    f_v       = grid.f_v

    ρ1   = FT(p.rho1)
    ρ2   = FT(p.rho2)
    hmin = FT(p.hmin)
    g    = FT(p.g)
    gp   = FT(p.gp)
    ν    = FT(p.nu)
    dt   = FT(p.dt)

    # AM3 weights
    Wnp1 = FT(5) / FT(12)   # weight for q^{n+1,*}
    Wn   = FT(2) / FT(3)    # weight for q^n
    Wnm1 = -FT(1) / FT(12)  # weight for q^{n-1}

    # --------------------------------------------------------
    # Prognostic variables (current time n)
    # --------------------------------------------------------
    h1 = prog.h1
    h2 = prog.h2
    H  = prog.H

    m1 = prog.m1
    m2 = prog.m2
    M  = prog.M

    n1 = prog.n1
    n2 = prog.n2
    N  = prog.N

    # Update total (barotropic) fields at n
    @. H = h1 + h2
    @. M = m1 + m2
    @. N = n1 + n2

    # Active layer (current time n)
    h_n = (layer == 1) ? h1 : h2
    m_n = (layer == 1) ? m1 : m2
    n_n = (layer == 1) ? n1 : n2

    # Histories: q^{n-1} from hist.*_tm0
    h_nm1_hist = (layer == 1) ? hist.h1_tm0 : hist.h2_tm0
    m_nm1_hist = (layer == 1) ? hist.m1_tm0 : hist.m2_tm0
    n_nm1_hist = (layer == 1) ? hist.n1_tm0 : hist.n2_tm0

    # For step 1, fall back to q^{n-1} ≡ q^n
    h_nm1 = isFirstTimeStep ? h_n : h_nm1_hist
    m_nm1 = isFirstTimeStep ? m_n : m_nm1_hist
    n_nm1 = isFirstTimeStep ? n_n : n_nm1_hist

    # Interpolated thickness fields at time n (updated below)
    H_in_u  = intp.H_in_u
    H_in_v  = intp.H_in_v
    h1_in_u = intp.h1_in_u
    h1_in_v = intp.h1_in_v
    h2_in_u = intp.h2_in_u
    h2_in_v = intp.h2_in_v

    # Star (time-centered) intermediates
    h1_star = intm.h1_star
    h2_star = intm.h2_star
    H_star  = intm.H_star

    m1_star = intm.m1_star
    m2_star = intm.m2_star
    M_star  = intm.M_star

    n1_star = intm.n1_star
    n2_star = intm.n2_star
    N_star  = intm.N_star

    # Active-layer star fields
    h_star = (layer == 1) ? h1_star : h2_star
    m_star = (layer == 1) ? m1_star : m2_star
    n_star = (layer == 1) ? n1_star : n2_star

    # Forcing
    taux_sf = forc.taux_sf
    tauy_sf = forc.tauy_sf
    taux_bt = forc.taux_bt
    tauy_bt = forc.tauy_bt

    # --------------------------------------------------------
    # 1. Apply wall BC to current fields and reconstruct h, H
    # --------------------------------------------------------
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n_n, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h_n, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(H,   Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h1_in_u, h1, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h1_in_v, h1, Nx, Ny, hmin)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h2_in_u, h2, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h2_in_v, h2, Nx, Ny, hmin)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(H_in_u, H, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(H_in_v, H, Nx, Ny, hmin)

    # Active layer thickness at faces
    h_in_u = (layer == 1) ? h1_in_u : h2_in_u
    h_in_v = (layer == 1) ? h1_in_v : h2_in_v

    # --------------------------------------------------------
    # 2. Build RHS R^n for momentum (m, n)
    # --------------------------------------------------------
    rhs_m = temp.temp_var_x1   # R(m)
    rhs_n = temp.temp_var_y1   # R(n)

    # 2a. Biharmonic viscosity: u = m/h_in_u, v = n/h_in_v
    u = temp.temp_var_x2
    v = temp.temp_var_y2
    @. u = m_n / h_in_u
    @. v = n_n / h_in_v

    biharmonic_m = temp.temp_var_x3
    biharmonic_n = temp.temp_var_y3

    calculate_biharmonic_term!(
        biharmonic_m, biharmonic_n,
        u, v,
        h_in_u, h_in_v,
        temp.temp_var_x2, temp.temp_var_y2,  # reuse as buffers
        grid, threads2, blocks2,
        ν, Nx, Ny,
    )

    @. rhs_m = biharmonic_m
    @. rhs_n = biharmonic_n

    # 2b. Surface / bottom stress (optional barotropic split)
    forc_m = temp.temp_var_x2
    forc_n = temp.temp_var_y2

    if mode_split
        if layer == 1
            @. forc_m =  taux_sf/ρ1 - h_in_u/H_in_u * (taux_sf/ρ1 + taux_bt/ρ2)
            @. forc_n =  tauy_sf/ρ1 - h_in_v/H_in_v * (tauy_sf/ρ1 + tauy_bt/ρ2)
        else
            @. forc_m =  taux_bt/ρ2 - h_in_u/H_in_u * (taux_sf/ρ1 + taux_bt/ρ2)
            @. forc_n =  tauy_bt/ρ2 - h_in_v/H_in_v * (tauy_sf/ρ1 + tauy_bt/ρ2)
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

    @. rhs_m += forc_m
    @. rhs_n += forc_n

    # 2c. Curvature + explicit Coriolis
    curv_x = temp.temp_var_x2
    curv_y = temp.temp_var_y2

    @cuda threads=threads2 blocks=blocks2 k_calc_curvature_terms!(
        curv_x, curv_y,
        m_n, n_n,
        h_in_u, h_in_v,
        lat_u, lat_v,
        Nx, Ny,
        FT(p.earthRadius),
    )

    @. rhs_m += curv_x
    @. rhs_n += curv_y

    # Coriolis using velocities reconstructed on cross faces
    u = temp.temp_var_x2
    v = temp.temp_var_y2
    @. u = m_n / h_in_u
    @. v = n_n / h_in_v

    u_in_v = temp.temp_var_x3
    v_in_u = temp.temp_var_y3

    @cuda threads=threads2 blocks=blocks2 k_recon_u_in_v!(u_in_v, u, Nx, Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_v_in_u!(v_in_u, v, Nx, Ny)

    @. rhs_m += h_in_u * f_u * v_in_u
    @. rhs_n -= h_in_v * f_v * u_in_v

    # 2d. WENO-Z advection of m and n
    adv_m = temp.temp_var_x2
    adv_n = temp.temp_var_y2

    # u-cells (advect m)
    u_face_for_u = temp.temp_var_x3
    v_face_for_u = temp.temp_var_y3

    @cuda threads=threads2 blocks=blocks2 k_calc_faceVels_for_ucell!(
        u_face_for_u, v_face_for_u,
        m_n, n_n,
        h_n,
        Nx, Ny,
        hmin,
    )
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(v_face_for_u, Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        adv_m,
        m_n,
        u_face_for_u, v_face_for_u,
        dx_face_u, dy_face_u,
        dArea_u,
        Nx, Ny,
        UGRID,
    )
    @. rhs_m += adv_m

    # v-cells (advect n)
    u_face_for_v = temp.temp_var_x3
    v_face_for_v = temp.temp_var_y3

    @cuda threads=threads2 blocks=blocks2 k_calc_faceVels_for_vcell!(
        u_face_for_v, v_face_for_v,
        m_n, n_n,
        h_n,
        Nx, Ny,
        hmin,
    )
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(v_face_for_v, Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        adv_n,
        n_n,
        u_face_for_v, v_face_for_v,
        dx_face_v, dy_face_v,
        dArea_v,
        Nx, Ny,
        VGRID,
    )
    @. rhs_n += adv_n

    # 2e. Pressure-gradient (baroclinic + optional barotropic)
    press_gradx = temp.temp_var_x2
    press_grady = temp.temp_var_y2

    h2_gradx = temp.temp_var_x3
    h2_grady = temp.temp_var_y3

    @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
        h2_gradx, h2_grady,
        h2,
        dx_n2n_h, dy_n2n_h,
        Nx, Ny,
    )

    if mode_split
        if layer == 1
            @. press_gradx =  gp * h1_in_u * h2_gradx
            @. press_grady =  gp * h1_in_v * h2_grady
        else
            @. press_gradx = -gp * h2_in_u * h2_gradx
            @. press_grady = -gp * h2_in_v * h2_grady
        end
    else
        H_gradx = temp.temp_var_x3
        H_grady = temp.temp_var_y3

        @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
            H_gradx, H_grady,
            H,
            dx_n2n_h, dy_n2n_h,
            Nx, Ny,
        )

        if layer == 1
            @. press_gradx = -g * h1_in_u * H_gradx
            @. press_grady = -g * h1_in_v * H_grady
        else
            @. press_gradx = -g * h2_in_u * H_gradx - gp * h2_in_u * h2_gradx
            @. press_grady = -g * h2_in_v * H_grady - gp * h2_in_v * h2_grady
        end
    end

    @. rhs_m += press_gradx
    @. rhs_n += press_grady

    # --------------------------------------------------------
    # 3. RHS R^n for thickness: h_t + ∇·(u h) = 0
    # --------------------------------------------------------
    rhs_h = temp.temp_var_x3

    u = temp.temp_var_x2
    v = temp.temp_var_y2
    @. u = m_n / h_in_u
    @. v = n_n / h_in_v

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        rhs_h,
        h_n,
        u, v,
        dx_face_h, dy_face_h,
        dArea_h,
        Nx, Ny,
        HGRID,
    )
    # rhs_h ~ −∇·(u h)

    # --------------------------------------------------------
    # 4. Leapfrog predictor: q^{n+1,*} = q^{n-1} + 2Δt R^n
    # --------------------------------------------------------
    # @. m_star = m_nm1 + 2f0(dt) * rhs_m
    # @. n_star = n_nm1 + 2f0(dt) * rhs_n
    # @. h_star = h_nm1 + 2f0(dt) * rhs_h

    @. m_star = m_n + FT(dt) * rhs_m
    @. n_star = n_n + FT(dt) * rhs_n
    @. h_star = h_n + FT(dt) * rhs_h
    

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n_star, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h_star, Nx, Ny)

    # --------------------------------------------------------
    # 5. AM3 time-centering to get q^{n+1/2}
    # --------------------------------------------------------
    # @. m_star = Wnp1 * m_star + Wn * m_n + Wnm1 * m_nm1
    # @. n_star = Wnp1 * n_star + Wn * n_n + Wnm1 * n_nm1
    # @. h_star = Wnp1 * h_star + Wn * h_n + Wnm1 * h_nm1

    # --------------------------------------------------------
    # 5. Heuns time-centering to get q^{n+1/2}
    # --------------------------------------------------------
    @. m_star = FT(0.5) * (m_star + m_n)
    @. n_star = FT(0.5) * (n_star + n_n)
    @. h_star = FT(0.5) * (h_star + h_n)

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n_star, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h_star, Nx, Ny)

    # Store into intm struct
    if layer == 1
        @. h1_star = h_star
        @. m1_star = m_star
        @. n1_star = n_star
    else
        @. h2_star = h_star
        @. m2_star = m_star
        @. n2_star = n_star
    end

    @. H_star = h1_star + h2_star
    @. M_star = m1_star + m2_star
    @. N_star = n1_star + n2_star

    return nothing
end


# ============================================================
# CORRECTOR: centered PGF + Coriolis + h-corrector
# ============================================================

"""
    corrector_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, p,
        threads1, blocks1,
        threads2, blocks2;
        layer      = 1,
        mode_split = true,
    )

Baroclinic **corrector** for a single layer (1 or 2).

On entry, `prog` holds q^n and `intm` holds the AM3-centered star
fields, interpreted as q^{n+1/2}.  This routine:

  1. Computes centered RHS R(q^{n+1/2}) from the star fields.
  2. Updates momentum:
         m^{n+1} = m^n + Δt R_m(q^{n+1/2})
         n^{n+1} = n^n + Δt R_n(q^{n+1/2})
  3. Updates thickness:
         h^{n+1} = h^n + Δt (−∇·(u^{n+1/2} h^{n+1/2}))
"""
function corrector_baroclinic!(
    prog::Prognostic,
    hist::history,
    intm::intermediate,
    temp::Temporary,
    intp::interpolated,
    forc::Forcing,
    grid::Grid,
    p::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int};
    layer::Int=1,
    mode_split::Bool=true,
)
    # --------------------------------------------------------
    # Grid & scalar aliases
    # --------------------------------------------------------
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
    dx_n2n_h  = grid.dx_n2n_h
    dy_n2n_h  = grid.dy_n2n_h
    f_u       = grid.f_u
    f_v       = grid.f_v

    ρ1   = FT(p.rho1)
    ρ2   = FT(p.rho2)
    hmin = FT(p.hmin)
    g    = FT(p.g)
    gp   = FT(p.gp)
    ν    = FT(p.nu)
    dt   = FT(p.dt)

    # --------------------------------------------------------
    # Prognostic fields at time n (input) and star fields
    # --------------------------------------------------------
    h1 = prog.h1
    h2 = prog.h2
    H  = prog.H

    m1 = prog.m1
    m2 = prog.m2
    M  = prog.M

    n1 = prog.n1
    n2 = prog.n2
    N  = prog.N

    @. H = h1 + h2
    @. M = m1 + m2
    @. N = n1 + n2

    # Star (time-centered) fields (from predictor)
    h1_star = intm.h1_star
    h2_star = intm.h2_star
    H_star  = intm.H_star

    m1_star = intm.m1_star
    m2_star = intm.m2_star
    M_star  = intm.M_star

    n1_star = intm.n1_star
    n2_star = intm.n2_star
    N_star  = intm.N_star

    # Active-layer: q^n and q^{n+1/2}
    h_n = (layer == 1) ? h1 : h2
    m_n = (layer == 1) ? m1 : m2
    n_n = (layer == 1) ? n1 : n2

    h_c = (layer == 1) ? h1_star : h2_star
    m_c = (layer == 1) ? m1_star : m2_star
    n_c = (layer == 1) ? n1_star : n2_star

    # Interpolated thickness at faces for centered fields
    Hc_in_u  = intp.H_in_u
    Hc_in_v  = intp.H_in_v
    h1c_in_u = intp.h1_in_u
    h1c_in_v = intp.h1_in_v
    h2c_in_u = intp.h2_in_u
    h2c_in_v = intp.h2_in_v

    hc_in_u = (layer == 1) ? h1c_in_u : h2c_in_u
    hc_in_v = (layer == 1) ? h1c_in_v : h2c_in_v

    # Forcing
    taux_sf = forc.taux_sf
    tauy_sf = forc.tauy_sf
    taux_bt = forc.taux_bt
    tauy_bt = forc.tauy_bt

    # --------------------------------------------------------
    # 1. Apply wall BC to centered fields and reconstruct h,H
    # --------------------------------------------------------
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n_c, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h_c, Nx, Ny)

    @. H_star = h1_star + h2_star
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(H_star, Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h1c_in_u, h1_star, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h1c_in_v, h1_star, Nx, Ny, hmin)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h2c_in_u, h2_star, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h2c_in_v, h2_star, Nx, Ny, hmin)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(Hc_in_u, H_star, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(Hc_in_v, H_star, Nx, Ny, hmin)

    # --------------------------------------------------------
    # 2. Build centered RHS R(q^{n+1/2}) for momentum
    # --------------------------------------------------------
    rhs_m = temp.temp_var_x1
    rhs_n = temp.temp_var_y1

    # 2a. Biharmonic viscosity with centered velocities
    uc = temp.temp_var_x2
    vc = temp.temp_var_y2
    @. uc = m_c / hc_in_u
    @. vc = n_c / hc_in_v

    biharmonic_m = temp.temp_var_x3
    biharmonic_n = temp.temp_var_y3

    calculate_biharmonic_term!(
        biharmonic_m, biharmonic_n,
        uc, vc,
        hc_in_u, hc_in_v,
        temp.temp_var_x2, temp.temp_var_y2,
        grid, threads2, blocks2,
        ν, Nx, Ny,
    )

    @. rhs_m = biharmonic_m
    @. rhs_n = biharmonic_n

    # 2b. Centered stresses
    forc_m = temp.temp_var_x2
    forc_n = temp.temp_var_y2

    if mode_split
        if layer == 1
            @. forc_m =  taux_sf/ρ1 - hc_in_u/Hc_in_u * (taux_sf/ρ1 + taux_bt/ρ2)
            @. forc_n =  tauy_sf/ρ1 - hc_in_v/Hc_in_v * (tauy_sf/ρ1 + tauy_bt/ρ2)
        else
            @. forc_m =  taux_bt/ρ2 - hc_in_u/Hc_in_u * (taux_sf/ρ1 + taux_bt/ρ2)
            @. forc_n =  tauy_bt/ρ2 - hc_in_v/Hc_in_v * (tauy_sf/ρ1 + tauy_bt/ρ2)
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

    @. rhs_m += forc_m
    @. rhs_n += forc_n

    # 2c. Curvature + explicit Coriolis (centered)
    curv_x = temp.temp_var_x2
    curv_y = temp.temp_var_y2

    @cuda threads=threads2 blocks=blocks2 k_calc_curvature_terms!(
        curv_x, curv_y,
        m_c, n_c,
        hc_in_u, hc_in_v,
        lat_u, lat_v,
        Nx, Ny,
        FT(p.earthRadius),
    )

    @. rhs_m += curv_x
    @. rhs_n += curv_y

    uc = temp.temp_var_x2
    vc = temp.temp_var_y2
    @. uc = m_c / hc_in_u
    @. vc = n_c / hc_in_v

    uc_in_v = temp.temp_var_x3
    vc_in_u = temp.temp_var_y3

    @cuda threads=threads2 blocks=blocks2 k_recon_u_in_v!(uc_in_v, uc, Nx, Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_v_in_u!(vc_in_u, vc, Nx, Ny)

    @. rhs_m += hc_in_u * f_u * vc_in_u
    @. rhs_n -= hc_in_v * f_v * uc_in_v

    # 2d. WENO-Z advection of centered m, n
    adv_m = temp.temp_var_x2
    adv_n = temp.temp_var_y2

    uc_face_for_u = temp.temp_var_x3
    vc_face_for_u = temp.temp_var_y3

    @cuda threads=threads2 blocks=blocks2 k_calc_faceVels_for_ucell!(
        uc_face_for_u, vc_face_for_u,
        m_c, n_c,
        h_c,
        Nx, Ny,
        hmin,
    )
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(vc_face_for_u, Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        adv_m,
        m_c,
        uc_face_for_u, vc_face_for_u,
        dx_face_u, dy_face_u,
        dArea_u,
        Nx, Ny,
        UGRID,
    )
    @. rhs_m += adv_m

    uc_face_for_v = temp.temp_var_x3
    vc_face_for_v = temp.temp_var_y3

    @cuda threads=threads2 blocks=blocks2 k_calc_faceVels_for_vcell!(
        uc_face_for_v, vc_face_for_v,
        m_c, n_c,
        h_c,
        Nx, Ny,
        hmin,
    )
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(vc_face_for_v, Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        adv_n,
        n_c,
        uc_face_for_v, vc_face_for_v,
        dx_face_v, dy_face_v,
        dArea_v,
        Nx, Ny,
        VGRID,
    )
    @. rhs_n += adv_n

    # 2e. Centered PGF using h2_star and H_star
    press_gradx = temp.temp_var_x2
    press_grady = temp.temp_var_y2

    h2_gradx = temp.temp_var_x3
    h2_grady = temp.temp_var_y3

    @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
        h2_gradx, h2_grady,
        h2_star,
        dx_n2n_h, dy_n2n_h,
        Nx, Ny,
    )

    if mode_split
        if layer == 1
            @. press_gradx =  gp * h1c_in_u * h2_gradx
            @. press_grady =  gp * h1c_in_v * h2_grady
        else
            @. press_gradx = -gp * h2c_in_u * h2_gradx
            @. press_grady = -gp * h2c_in_v * h2_grady
        end
    else
        H_gradx = temp.temp_var_x3
        H_grady = temp.temp_var_y3

        @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
            H_gradx, H_grady,
            H_star,
            dx_n2n_h, dy_n2n_h,
            Nx, Ny,
        )

        if layer == 1
            @. press_gradx = -g * h1c_in_u * H_gradx
            @. press_grady = -g * h1c_in_v * H_grady
        else
            @. press_gradx = -g * h2c_in_u * H_gradx - gp * h2c_in_u * h2_gradx
            @. press_grady = -g * h2c_in_v * H_grady - gp * h2c_in_v * h2_grady
        end
    end

    @. rhs_m += press_gradx
    @. rhs_n += press_grady

    # --------------------------------------------------------
    # 3. Momentum corrector: m^{n+1} = m^n + Δt R_m, etc.
    # --------------------------------------------------------
    @. m_n = m_n + dt * rhs_m
    @. n_n = n_n + dt * rhs_n

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n_n, Nx, Ny)

    if layer == 1
        prog.m1 .= m_n
        prog.n1 .= n_n
    else
        prog.m2 .= m_n
        prog.n2 .= n_n
    end

    @. prog.M = prog.m1 + prog.m2
    @. prog.N = prog.n1 + prog.n2

    # --------------------------------------------------------
    # 4. Thickness corrector using centered velocities
    # --------------------------------------------------------
    uc = temp.temp_var_x2
    vc = temp.temp_var_y2
    @. uc = m_c / hc_in_u
    @. vc = n_c / hc_in_v

    minus_div_uh = temp.temp_var_x3

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        minus_div_uh,
        h_c,
        uc, vc,
        dx_face_h, dy_face_h,
        dArea_h,
        Nx, Ny,
        HGRID,
    )

    @. h_n = h_n + dt * minus_div_uh
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h_n, Nx, Ny)

    if layer == 1
        prog.h1 .= h_n
    else
        prog.h2 .= h_n
    end

    @. prog.H = prog.h1 + prog.h2

    return nothing
end


# ============================================================
# Public API: baroclinic step for both layers
# ============================================================

"""
    step_baroclinic!(
        state, grid, p;
        threads1, blocks1,
        threads2, blocks2,
        step,
        mode_split = true,
    )

Advance both baroclinic layers (1 and 2) by one baroclinic time step
`p.dt` using the LF–AM3 scheme:

  * First step (`step == 1`):
      initialize 3-level history: q_tm0 = q_tm1 = q^0

  * Predictor (per layer):
      build R(q^n), leapfrog to q^{n+1,*}, AM3 to q^{n+1/2}

  * Corrector (per layer):
      rebuild R from q^{n+1/2}, update q^{n+1} with Euler

  * Rotate history: (n−1, n) ← (old n, new n+1)
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
    prog = state.prog
    intp = state.intp
    intm = state.intm
    hist = state.hist
    temp = state.temp
    forc = state.forc

    # Initialize history at first baroclinic step
    if step == 1
        initialize_baroclinic_history!(hist, prog)
    end

    # Predictor: layer 1 & 2
    predictor_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, p,
        threads1, blocks1,
        threads2, blocks2;
        isFirstTimeStep = (step == 1),
        layer           = 1,
        mode_split      = mode_split,
    )

    predictor_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, p,
        threads1, blocks1,
        threads2, blocks2;
        isFirstTimeStep = (step == 1),
        layer           = 2,
        mode_split      = mode_split,
    )

    # Corrector: layer 1 & 2
    corrector_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, p,
        threads1, blocks1,
        threads2, blocks2;
        layer      = 1,
        mode_split = mode_split,
    )

    corrector_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, p,
        threads1, blocks1,
        threads2, blocks2;
        layer      = 2,
        mode_split = mode_split,
    )

    # Rotate LF–AM3 history
    rotate_baroclinic_history!(hist, prog)

    return nothing
end