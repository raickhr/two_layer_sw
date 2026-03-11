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
# using a ROMS-style AB3–AM3 predictor–corrector scheme over
# one baroclinic time step p.dt:
#
#   1. Predictor (AB3, explicit):
#        q^{n+1,pre} = q^n
#                      + Δt [ 23/12 R(q^n)
#                              −16/12 R(q^{n−1})
#                               5/12 R(q^{n−2}) ]
#
#   2. Corrector (AM3, implicit in the tendencies):
#        q^{n+1} = q^n
#                  + Δt [  5/12 R(q^{n+1,pre})
#                          8/12 R(q^n)
#                         −1/12 R(q^{n−1}) ]
#
# where q stands for (m, n, h) for each layer.
#
# prog::Prognostic holds the state at the *current* time n and
# is updated in-place to time n+1.
#
# hist::History stores *tendency histories*:
#   rhs_*_tm0 ≈ R(q^n)
#   rhs_*_tm1 ≈ R(q^{n−1})
#   rhs_*_tm2 ≈ R(q^{n−2})
#   rhs_*_tp1 ≈ R(q^{n+1}, pre)   
#
# intm::Intermediate holds the provisional AB3-predicted fields:
#   h1_star, h2_star, H_star
#   m1_star, m2_star, M_star
#   n1_star, n2_star, N_star
############################################################


# ============================================================
# Helper: rotate 3-level RHS History (baroclinic part)
# ============================================================

function rotate_baroclinic_rhs_History!(hist::History)
    # Layer 1
    @. hist.rhs_m1_tm2 = hist.rhs_m1_tm1
    @. hist.rhs_m1_tm1 = hist.rhs_m1_tm0
    @. hist.rhs_m1_tm0 = hist.rhs_m1_tp1

    @. hist.rhs_n1_tm2 = hist.rhs_n1_tm1
    @. hist.rhs_n1_tm1 = hist.rhs_n1_tm0
    @. hist.rhs_n1_tm0 = hist.rhs_n1_tp1

    @. hist.rhs_h1_tm2 = hist.rhs_h1_tm1
    @. hist.rhs_h1_tm1 = hist.rhs_h1_tm0
    @. hist.rhs_h1_tm0 = hist.rhs_h1_tp1

    # Layer 2
    @. hist.rhs_m2_tm2 = hist.rhs_m2_tm1
    @. hist.rhs_m2_tm1 = hist.rhs_m2_tm0
    @. hist.rhs_m2_tm0 = hist.rhs_m2_tp1

    @. hist.rhs_n2_tm2 = hist.rhs_n2_tm1
    @. hist.rhs_n2_tm1 = hist.rhs_n2_tm0
    @. hist.rhs_n2_tm0 = hist.rhs_n2_tp1

    @. hist.rhs_h2_tm2 = hist.rhs_h2_tm1
    @. hist.rhs_h2_tm1 = hist.rhs_h2_tm0
    @. hist.rhs_h2_tm0 = hist.rhs_h2_tp1

    return nothing
end


# ============================================================
# Core tendency builder
# ============================================================

"""
    compute_baroclinic_rhs!(
        rhs_m, rhs_n, rhs_h,
        h1, h2, H,
        m1, m2, n1, n2,
        intp, temp, forc, grid, p,
        threads1, blocks1, threads2, blocks2;
        layer::Int=1, mode_split::Bool=true,
    )

Compute the baroclinic right-hand sides R = (R_m, R_n, R_h) for a
single layer (1 or 2), given the full two-layer state (h1, h2,
H, m1, m2, n1, n2) at some time level.

This routine is *stateless*: it does not change `prog` or `hist`
and can be reused for:
  - R(q^n)           (current state)
  - R(q^{n+1,pre})   (AB3 predictor)
  - R(q^{n+1})       (final state, for History rotation)
"""
function compute_baroclinic_rhs!(
    rhs_m::CuArray{FT,2},
    rhs_n::CuArray{FT,2},
    rhs_h::CuArray{FT,2},
    h1::CuArray{FT,2},
    h2::CuArray{FT,2},
    H::CuArray{FT,2},
    m1::CuArray{FT,2},
    m2::CuArray{FT,2},
    n1::CuArray{FT,2},
    n2::CuArray{FT,2},
    intp::Interpolated,
    temp::Temporary,
    debug::Debug,
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

    ρ1   = FT(p.rho1)
    ρ2   = FT(p.rho2)
    hmin = FT(p.hmin)
    g    = FT(p.g)
    gp   = FT(p.gp)
    ν    = FT(p.nu)

    # --------------------------------------------------------
    # Active-layer aliases at this time level
    # --------------------------------------------------------
    h = (layer == 1) ? h1 : h2
    m = (layer == 1) ? m1 : m2
    n = (layer == 1) ? n1 : n2

    # Apply wall BCs to prognostic fields (in-place)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n,  Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h,  Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(H,  Nx, Ny)

    # Interpolated thickness at faces (reuse intp buffers)
    h1_in_u = intp.h1_in_u
    h1_in_v = intp.h1_in_v
    h2_in_u = intp.h2_in_u
    h2_in_v = intp.h2_in_v
    H_in_u  = intp.H_in_u
    H_in_v  = intp.H_in_v

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h1_in_u, h1, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h1_in_v, h1, Nx, Ny, hmin)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h2_in_u, h2, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h2_in_v, h2, Nx, Ny, hmin)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(H_in_u,  H,  Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(H_in_v,  H,  Nx, Ny, hmin)

    h_in_u = (layer == 1) ? h1_in_u : h2_in_u
    h_in_v = (layer == 1) ? h1_in_v : h2_in_v

    # Forcing
    taux_sf = forc.taux_sf
    tauy_sf = forc.tauy_sf
    taux_bt = forc.taux_bt
    tauy_bt = forc.tauy_bt

    # --------------------------------------------------------
    # 1. Momentum tendencies R_m, R_n
    # --------------------------------------------------------
    # 1a. Biharmonic viscosity using u = m/h_in_u, v = n/h_in_v
    u = temp.temp_var_x2
    v = temp.temp_var_y2
    @. u = m / h_in_u
    @. v = n / h_in_v

    biharmonic_m = temp.temp_var_x3
    biharmonic_n = temp.temp_var_y3

    buf_x = temp.temp_var_x4
    buf_y = temp.temp_var_y4

    calculate_biharmonic_term!(
        biharmonic_m, biharmonic_n,
        u, v,
        h_in_u, h_in_v,
        buf_x, buf_y,
        grid, threads2, blocks2,
        ν, Nx, Ny,
    )

    @. rhs_m = biharmonic_m
    @. rhs_n = biharmonic_n

    # 1b. Surface / bottom stress (optional barotropic split)
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
            @. forc_m =  taux_sf/ρ1
            @. forc_n =  tauy_sf/ρ1
        else
            @. forc_m =  taux_bt/ρ2
            @. forc_n =  tauy_bt/ρ2
        end
    end

    @. rhs_m += forc_m
    @. rhs_n += forc_n

    # 1c. Curvature (metric) terms
    curv_x = temp.temp_var_x2
    curv_y = temp.temp_var_y2

    @cuda threads=threads2 blocks=blocks2 k_calc_curvature_terms!(
        curv_x, curv_y,
        m, n,
        h_in_u, h_in_v,
        lat_u, lat_v,
        Nx, Ny,
        FT(p.earthRadius),
    )

    @. rhs_m += curv_x
    @. rhs_n += curv_y

    # 1d. Coriolis terms
    cor_x = temp.temp_var_x2
    cor_y = temp.temp_var_y2

    u_in_v = temp.temp_var_x4
    v_in_u = temp.temp_var_y4

    @. u = m / h_in_u
    @. v = n / h_in_v

    @cuda threads=threads1 blocks=blocks1 k_recon_u_in_v!(u_in_v, u, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_recon_v_in_u!(v_in_u, v, Nx, Ny)

    @. cor_x =   h_in_u * grid.f_u * v_in_u
    @. cor_y =  -h_in_v * grid.f_v * u_in_v

    @. rhs_m += cor_x
    @. rhs_n += cor_y

    # 1e. WENO-Z advection of m and n
    adv_m = temp.temp_var_x2
    adv_n = temp.temp_var_y2

    # u-cells (advect m)
    u_face_for_u = temp.temp_var_x3
    v_face_for_u = temp.temp_var_y3

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
    @. rhs_m += adv_m

    # v-cells (advect n)
    u_face_for_v = temp.temp_var_x3
    v_face_for_v = temp.temp_var_y3

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
    @. rhs_n += adv_n

    # 1f. Pressure-gradient (baroclinic + optional barotropic)
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
            @. press_gradx = -gp * h1_in_u * h2_gradx
            @. press_grady = -gp * h1_in_v * h2_grady
        else
            @. press_gradx =  gp * h2_in_u * h2_gradx
            @. press_grady =  gp * h2_in_v * h2_grady
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
    # 2. Thickness tendency R_h: h_t + ∇·(u h) = 0
    # --------------------------------------------------------
    @. u = m / h_in_u
    @. v = n / h_in_v

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        rhs_h,
        h,
        u, v,
        dx_face_h, dy_face_h,
        dArea_h,
        Nx, Ny,
        HGRID,
    )
    # rhs_h ≈ −∇·(u h)

    return nothing
end


# ============================================================
# PREDICTOR: AB3 to get provisional q^{n+1,pre}
# ============================================================

"""
    predictor_baroclinic!(...)

AB3 predictor for a single baroclinic layer (1 or 2).

On entry:
  - prog contains q^n
  - hist.rhs_*_{tm0,tm1,tm2} contain R^n, R^{n−1}, R^{n−2}
    (for step = 1 or 2 these are effectively bootstrapped).

On exit:
  - intm.{h*,m*,n*}_star contain q^{n+1,pre}
  - hist.rhs_*_tm0 is overwritten with freshly computed R^n
"""
function predictor_baroclinic!(
    prog::Prognostic,
    hist::History,
    intm::Intermediate,
    temp::Temporary,
    intp::Interpolated,
    forc::Forcing,
    grid::Grid,
    p::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int};
    layer::Int=1,
    step::Int=1,
    mode_split::Bool=true,
)
    dt = FT(p.dt)

    # Prognostic aliases at time n
    h1 = prog1.h
    h2 = prog2.h
    H  = prog.H

    m1 = prog1.m
    m2 = prog2.m
    n1 = prog1.n
    n2 = prog2.n

    # Active-layer fields at time n
    h_n = (layer == 1) ? h1 : h2
    m_n = (layer == 1) ? m1 : m2
    n_n = (layer == 1) ? n1 : n2

    # RHS History views for this layer
    rhs_m_tm0 = (layer == 1) ? hist.rhs_m1_tm0 : hist.rhs_m2_tm0
    rhs_m_tm1 = (layer == 1) ? hist.rhs_m1_tm1 : hist.rhs_m2_tm1
    rhs_m_tm2 = (layer == 1) ? hist.rhs_m1_tm2 : hist.rhs_m2_tm2

    rhs_n_tm0 = (layer == 1) ? hist.rhs_n1_tm0 : hist.rhs_n2_tm0
    rhs_n_tm1 = (layer == 1) ? hist.rhs_n1_tm1 : hist.rhs_n2_tm1
    rhs_n_tm2 = (layer == 1) ? hist.rhs_n1_tm2 : hist.rhs_n2_tm2

    rhs_h_tm0 = (layer == 1) ? hist.rhs_h1_tm0 : hist.rhs_h2_tm0
    rhs_h_tm1 = (layer == 1) ? hist.rhs_h1_tm1 : hist.rhs_h2_tm1
    rhs_h_tm2 = (layer == 1) ? hist.rhs_h1_tm2 : hist.rhs_h2_tm2

    # --------------------------------------------------------
    # 1. Compute R^n and store in rhs_*_tm0
    # --------------------------------------------------------
    compute_baroclinic_rhs!(
        rhs_m_tm0, rhs_n_tm0, rhs_h_tm0,
        h1, h2, H,
        m1, m2, n1, n2,
        intp, temp, debug, forc, grid, p,
        threads1, blocks1, threads2, blocks2;
        layer      = layer,
        mode_split = mode_split,
    )

    # Local aliases for the three RHS levels used in AB3
    Rm_n   = rhs_m_tm0
    Rn_n   = rhs_n_tm0
    Rh_n   = rhs_h_tm0

    Rm_nm1 = (step == 1) ? rhs_m_tm0 : rhs_m_tm1
    Rn_nm1 = (step == 1) ? rhs_n_tm0 : rhs_n_tm1
    Rh_nm1 = (step == 1) ? rhs_h_tm0 : rhs_h_tm1

    Rm_nm2 = (step <= 2) ? Rm_nm1 : rhs_m_tm2
    Rn_nm2 = (step <= 2) ? Rn_nm1 : rhs_n_tm2
    Rh_nm2 = (step <= 2) ? Rh_nm1 : rhs_h_tm2

    # AB3 coefficients
    c0 = FT(23) / FT(12)
    c1 = -FT(16) / FT(12)
    c2 =  FT(5)  / FT(12)

    # Star fields (provisional q^{n+1,pre})
    h1_star = intm.h1_star
    h2_star = intm.h2_star
    m1_star = intm.m1_star
    m2_star = intm.m2_star
    n1_star = intm.n1_star
    n2_star = intm.n2_star

    h_star = (layer == 1) ? h1_star : h2_star
    m_star = (layer == 1) ? m1_star : m2_star
    n_star = (layer == 1) ? n1_star : n2_star

    # --------------------------------------------------------
    # 2. AB3 update to get q^{n+1,pre}
    # --------------------------------------------------------
    @. m_star = m_n + dt * (c0 * Rm_n + c1 * Rm_nm1 + c2 * Rm_nm2)
    @. n_star = n_n + dt * (c0 * Rn_n + c1 * Rn_nm1 + c2 * Rn_nm2)
    @. h_star = h_n + dt * (c0 * Rh_n + c1 * Rh_nm1 + c2 * Rh_nm2)

    # Reapply wall BCs to provisional fields
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n_star,  Int(p.Nx), Int(p.Ny))
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h_star,  Int(p.Nx), Int(p.Ny))

    return nothing
end


# ============================================================
# CORRECTOR: AM3 using R^{n+1,pre}, R^n, R^{n−1}
# ============================================================

"""
    corrector_baroclinic!(...)

AM3 corrector for a single baroclinic layer (1 or 2).

On entry:
  - prog contains q^n
  - intm.*_star contain q^{n+1,pre} from AB3
  - hist.rhs_*_tm0,tm1 contain R^n and R^{n−1}

On exit:
  - prog fields for this layer are updated to q^{n+1}
  - hist.rhs_*_tp1 is *not* updated here; that is done after
    both layers are stepped, using the final q^{n+1}.
"""
function corrector_baroclinic!(
    prog::Prognostic,
    hist::History,
    intm::Intermediate,
    temp::Temporary,
    intp::Interpolated,
    forc::Forcing,
    grid::Grid,
    p::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int};
    layer::Int=1,
    step::Int=1,
    mode_split::Bool=true,
)
    dt = FT(p.dt)

    # Current-time fields q^n
    h1 = prog1.h
    h2 = prog2.h
    H  = prog.H

    m1 = prog1.m
    m2 = prog2.m
    n1 = prog1.n
    n2 = prog2.n

    h_n = (layer == 1) ? h1 : h2
    m_n = (layer == 1) ? m1 : m2
    n_n = (layer == 1) ? n1 : n2

    # Provisional fields q^{n+1,pre}
    h1_star = intm.h1_star
    h2_star = intm.h2_star
    H_star  = intm.H_star

    m1_star = intm.m1_star
    m2_star = intm.m2_star
    n1_star = intm.n1_star
    n2_star = intm.n2_star

    h_star = (layer == 1) ? h1_star : h2_star
    m_star = (layer == 1) ? m1_star : m2_star
    n_star = (layer == 1) ? n1_star : n2_star

    # RHS History views for this layer
    rhs_m_tp1 = (layer == 1) ? hist.rhs_m1_tp1 : hist.rhs_m2_tp1
    rhs_m_tm0 = (layer == 1) ? hist.rhs_m1_tm0 : hist.rhs_m2_tm0
    rhs_m_tm1 = (layer == 1) ? hist.rhs_m1_tm1 : hist.rhs_m2_tm1

    rhs_n_tp1 = (layer == 1) ? hist.rhs_n1_tp1 : hist.rhs_n2_tp1
    rhs_n_tm0 = (layer == 1) ? hist.rhs_n1_tm0 : hist.rhs_n2_tm0
    rhs_n_tm1 = (layer == 1) ? hist.rhs_n1_tm1 : hist.rhs_n2_tm1

    rhs_h_tp1 = (layer == 1) ? hist.rhs_h1_tp1 : hist.rhs_h2_tp1
    rhs_h_tm0 = (layer == 1) ? hist.rhs_h1_tm0 : hist.rhs_h2_tm0
    rhs_h_tm1 = (layer == 1) ? hist.rhs_h1_tm1 : hist.rhs_h2_tm1

    # --------------------------------------------------------
    # 1. Compute R(q^{n+1,pre}) into rhs_*_tp1
    # --------------------------------------------------------
    compute_baroclinic_rhs!(
        rhs_m_tp1, rhs_n_tp1, rhs_h_tp1,
        h1_star, h2_star, H_star,
        m1_star, m2_star, n1_star, n2_star,
        intp, temp, debug, forc, grid, p,
        threads1, blocks1, threads2, blocks2;
        layer      = layer,
        mode_split = mode_split,
    )

    # Local aliases for AM3 combination
    Rm_np1 = rhs_m_tp1
    Rn_np1 = rhs_n_tp1
    Rh_np1 = rhs_h_tp1

    Rm_n   = rhs_m_tm0
    Rn_n   = rhs_n_tm0
    Rh_n   = rhs_h_tm0

    Rm_nm1 = (step == 1) ? Rm_n : rhs_m_tm1
    Rn_nm1 = (step == 1) ? Rn_n : rhs_n_tm1
    Rh_nm1 = (step == 1) ? Rh_n : rhs_h_tm1

    # AM3 coefficients
    a0 = FT(5) / FT(12)
    a1 = FT(8) / FT(12)
    a2 = -FT(1) / FT(12)

    # --------------------------------------------------------
    # 2. AM3 update: q^{n+1}
    # --------------------------------------------------------
    @. m_n = m_n + dt * (a0 * Rm_np1 + a1 * Rm_n + a2 * Rm_nm1)
    @. n_n = n_n + dt * (a0 * Rn_np1 + a1 * Rn_n + a2 * Rn_nm1)
    @. h_n = h_n + dt * (a0 * Rh_np1 + a1 * Rh_n + a2 * Rh_nm1)

    # Enforce boundary conditions on updated fields
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n_n,  Int(p.Nx), Int(p.Ny))
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h_n,  Int(p.Nx), Int(p.Ny))

    # Write back to prog
    if layer == 1
        prog1.m .= m_n
        prog1.n .= n_n
        prog1.h .= h_n
    else
        prog2.m .= m_n
        prog2.n .= n_n
        prog2.h .= h_n
    end

    return nothing
end


# ============================================================
# Public API: baroclinic step for both layers
# ============================================================

"""
    step_baroclinic!(...)

Advance both baroclinic layers (1 and 2) by one baroclinic time step
`p.dt` using the AB3–AM3 predictor–corrector scheme.
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

    # Ensure barotropic totals at time n (before predictor)
    @. prog.H = prog1.h + prog2.h
    @. prog.M = prog1.m + prog2.m
    @. prog.N = prog1.n + prog2.n

    # Predictor (AB3): layer 1 & 2
    predictor_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, p,
        threads1, blocks1,
        threads2, blocks2;
        layer      = 1,
        step       = step,
        mode_split = mode_split,
    )

    predictor_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, p,
        threads1, blocks1,
        threads2, blocks2;
        layer      = 2,
        step       = step,
        mode_split = mode_split,
    )

    # Build provisional barotropic fields at n+1,pre
    @. intm.H_star = intm.h1_star + intm.h2_star
    @. intm.M_star = intm.m1_star + intm.m2_star
    @. intm.N_star = intm.n1_star + intm.n2_star

    # Corrector (AM3): layer 1 & 2
    corrector_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, p,
        threads1, blocks1,
        threads2, blocks2;
        layer      = 1,
        step       = step,
        mode_split = mode_split,
    )

    corrector_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, p,
        threads1, blocks1,
        threads2, blocks2;
        layer      = 2,
        step       = step,
        mode_split = mode_split,
    )

    # Update barotropic totals at new time n+1
    @. prog.H = prog1.h + prog2.h
    @. prog.M = prog1.m + prog2.m
    @. prog.N = prog1.n + prog2.n

    # --------------------------------------------------------
    # Recompute RHS at final time level q^{n+1} for History
    # --------------------------------------------------------
    compute_baroclinic_rhs!(
        hist.rhs_m1_tp1, hist.rhs_n1_tp1, hist.rhs_h1_tp1,
        prog1.h, prog2.h, prog.H,
        prog1.m, prog2.m, prog1.n, prog2.n,
        intp, temp, debug, forc, grid, p,
        threads1, blocks1, threads2, blocks2;
        layer      = 1,
        mode_split = mode_split,
    )

    compute_baroclinic_rhs!(
        hist.rhs_m2_tp1, hist.rhs_n2_tp1, hist.rhs_h2_tp1,
        prog1.h, prog2.h, prog.H,
        prog1.m, prog2.m, prog1.n, prog2.n,
        intp, temp, debug, forc, grid, p,
        threads1, blocks1, threads2, blocks2;
        layer      = 2,
        mode_split = mode_split,
    )

    # Rotate RHS History so that:
    #   tm0 ← R^{n+1}, tm1 ← R^{n}, tm2 ← R^{n−1}
    rotate_baroclinic_rhs_History!(hist)

    return nothing
end