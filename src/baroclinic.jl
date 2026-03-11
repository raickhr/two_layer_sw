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
# using a ROMS-style AB3–AM3 predictor–corrector scheme:
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
# prog::Prognostic holds the state at time n and is updated
# in-place to time n+1.
#
# hist::History stores tendency histories:
#   rhs_*_tm0 ≈ R(q^n)
#   rhs_*_tm1 ≈ R(q^{n−1})
#   rhs_*_tm2 ≈ R(q^{n−2})
#   rhs_*_tp1 ≈ R(q^{n+1,pre})
#
# intm::Intermediate holds provisional AB3-predicted fields:
#   h1_star, h2_star
#   m1_star, m2_star
#   n1_star, n2_star
############################################################

function calc_pressure_all!(
    p1, p2, h1, h2,
    g::FT,
    rho1::FT,
    rho2::FT,
    threads1::Int,
    blocks1::Int,
    Nx::Int,
    Ny::Int,
)
    #gp = (rho2 - rho1)/rho1 * g

    @. p1 = g * (h1 + h2)
    @. p2 = g * h1 + rho2/rho1 * g * h2

    # this is not full pressure. This is just the coupling pressure
    # for layer 1
    #       1/ρ * h1∇p = gh1∇(h1+ h2)   
    #                  = ∇(1/2 g h1²) + gh1∇h2 
    # First part goes into advective term. The second part is coupling part.
    # @. p1 = g * h2
    # @. p2 = rho1/rho2 * g * h1

    return nothing
end


function compute_baroclinic_rhs!(
    rhs_m::CuArray{FT,2},
    rhs_n::CuArray{FT,2},
    rhs_h::CuArray{FT,2},
    m::CuArray{FT,2},
    n::CuArray{FT,2},
    h::CuArray{FT,2},
    diag::Diagnostic,
    temp::Temporary,
    intp::Interpolated,
    grid::Grid,
    params::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
)
    Nx = Int(params.Nx)
    Ny = Int(params.Ny)

    hmin = FT(params.hmin)
    deg2rad = FT(π) / FT(180)

    # Zero RHS before accumulation
    @. rhs_m = 0
    @. rhs_n = 0
    @. rhs_h = 0
    

    ####################################################################
    #    Diagnostic and intermediate terms used multiple times
    ####################################################################

    # Interpolated layer thickness on u/v points
    h_in_u = intp.h_in_u
    h_in_v = intp.h_in_v

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h_in_u, h, Nx, Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h_in_v, h, Nx, Ny)

    # Enforce thickness floor before dividing
    @. h_in_u = max(h_in_u, hmin)
    @. h_in_v = max(h_in_v, hmin)


    # Velocities on u/v points
    u = diag.u
    v = diag.v

    @. u = m / h_in_u
    @. v = n / h_in_v

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_u!(u, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(v, Nx, Ny)
    
    @. m = u * h_in_u
    @. n = v * h_in_v

    # Cross-grid velocities
    u_in_v = intp.u_in_v
    v_in_u = intp.v_in_u

    @cuda threads=threads2 blocks=blocks2 k_recon_u_in_v!(u_in_v, u, Nx, Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_v_in_u!(v_in_u, v, Nx, Ny)

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_u_half!(u_in_v, Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v_half!(v_in_u, Nx, Ny)

    # ∇u and ∇v
    gradx_u = diag.gradx_u # HGRID
    grady_u = diag.grady_u # ZGRID

    @cuda threads=threads2 blocks=blocks2 k_calc_gradient_u!(
        gradx_u, grady_u, u,
        grid.dx_n2n_u,
        grid.dy_n2n_u,
        Nx, Ny,
    )

    gradx_v = diag.gradx_v # ZGRID
    grady_v = diag.grady_v # HGRID

    @cuda threads=threads2 blocks=blocks2 k_calc_gradient_v!(
        gradx_v, grady_v, v,
        grid.dx_n2n_v, grid.dy_n2n_v,
        Nx, Ny,
    )

    # Relative vorticity ζ on ZGRID
    zeta = diag.zeta
    @. zeta = gradx_v - grady_u  # ζ = ∂v/∂x - ∂u/∂y

    # Leith viscosity (currently constant ν = params.nu)
    viscosity_in_u = diag.viscosity_u
    viscosity_in_v = diag.viscosity_v

    # If/when you restore Leith: call calculate_leith_viscosity! here.
    calculate_leith_viscosity!(viscosity_in_u, viscosity_in_v, zeta,
                            grid.dx_n2n_z, grid.dy_n2n_z, 
                            grid.dx_n2n_u, grid.dy_n2n_u, 
                            grid.dx_n2n_v, grid.dy_n2n_v,
                            temp.temp_var_x1, temp.temp_var_y1, 
                            temp.temp_var_x2, temp.temp_var_y2, 
                            params.nu, threads2, blocks2, Nx, Ny)

    # @. viscosity_in_u = params.nu
    # @. viscosity_in_v = params.nu

    ##########################################################
    #                  MOMENTUM CONSERVATION
    ##########################################################

    # ========================================================
    # 1. Pressure Gradient
    # ========================================================
    press_gradx = diag.gradx_pres
    press_grady = diag.grady_pres
    pressure    = diag.pressure

    dx_n2n_h = grid.dx_n2n_h
    dy_n2n_h = grid.dy_n2n_h

    @cuda threads=threads2 blocks=blocks2 k_calc_gradient_h!(
        press_gradx, press_grady,
        pressure,
        dx_n2n_h, dy_n2n_h,
        Nx, Ny,
    )

    @. rhs_m = rhs_m - press_gradx * h_in_u #1/ρ is alrealy taken care of in pressure calculation
    @. rhs_n = rhs_n - press_grady * h_in_v

    # ========================================================
    # 2. Viscous term (Laplacian/Leith + metric curvature)
    # ========================================================
    viscous_m = diag.viscous_term_u
    viscous_n = diag.viscous_term_v

    calculate_viscous_term!(
        viscous_m, viscous_n,
        viscosity_in_u, viscosity_in_v,
        h,
        gradx_u, grady_u,
        gradx_v, grady_v,
        grid.dx_face_u, grid.dy_face_u, grid.dArea_u,
        grid.dx_face_v, grid.dy_face_v, grid.dArea_v,
        threads2, blocks2,
        Nx, Ny;
        hmin = hmin,
    )

    curv_visc_m = diag.visx_curv
    curv_visc_n = diag.visy_curv

    gradx_v_in_u = temp.temp_var_x1
    gradx_u_in_v = temp.temp_var_y1

    # gradx_v is on ZGRID → interpolate to UGRID
    @cuda threads=threads2 blocks=blocks2 k_recon_z_in_u!(
        gradx_v_in_u, gradx_v, Nx, Ny,
    )

    # gradx_u is on HGRID → interpolate to VGRID
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(
        gradx_u_in_v, gradx_u, Nx, Ny,
    )

    @. curv_visc_m =  viscosity_in_u * h_in_u *
                      (-FT(2) * tan(grid.lat_u_2d * deg2rad) * gradx_v_in_u / params.earthRadius - 
                      u/(params.earthRadius^2 * cos(grid.lat_u_2d * deg2rad)))

    @. curv_visc_n = viscosity_in_v * h_in_v *
                     (FT(2) * tan(grid.lat_v_2d * deg2rad) * gradx_u_in_v / params.earthRadius - 
                      v/(params.earthRadius^2 * cos(grid.lat_v_2d * deg2rad)))

    @. rhs_m = rhs_m + viscous_m + curv_visc_m
    @. rhs_n = rhs_n + viscous_n + curv_visc_n

    # ========================================================
    # 3. Coriolis Terms
    # ========================================================
    cor_x = diag.coriolis_term_u
    cor_y = diag.coriolis_term_v

    @. cor_x =  h_in_u * grid.f_u * v_in_u
    @. cor_y = -h_in_v * grid.f_v * u_in_v

    @. rhs_m = rhs_m + cor_x
    @. rhs_n = rhs_n + cor_y

    # ========================================================
    # 4. Advection Terms
    # ========================================================
    adv_m = diag.advec_term_u
    adv_n = diag.advec_term_v

    @cuda threads=threads2 blocks=blocks2 k_calc_flux_m_for_ucell!(
        adv_m,               # zonal-momentum tendency at u-cells
        m, n,                # m = h u (UGRID), n = h v (VGRID)
        h,
        grid.dx_face_u, grid.dy_face_u,
        grid.dArea_u,
        Nx, Ny,
        params.g,
        hmin,
    )

    @cuda threads=threads2 blocks=blocks2 k_calc_flux_n_for_vcell!(
        adv_n,               # meridional-momentum tendency at v-cells
        m, n,
        h,
        grid.dx_face_v, grid.dy_face_v,
        grid.dArea_v,
        Nx, Ny,
        params.g,
        hmin,
    )

    curv_x = diag.advx_curv
    curv_y = diag.advy_curv

    lat_u_2d = grid.lat_u_2d
    lat_v_2d = grid.lat_v_2d

    @. curv_x =  h_in_u * u * v_in_u * tan(lat_u_2d * deg2rad) / params.earthRadius
    @. curv_y = -h_in_v * u_in_v^2 * tan(lat_v_2d * deg2rad) / params.earthRadius

    @. rhs_m = rhs_m + adv_m + curv_x
    @. rhs_n = rhs_n + adv_n + curv_y

    ##########################################################
    #                  MASS CONSERVATION
    ##########################################################
    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        rhs_h,
        h,
        u, v,
        grid.dx_face_h, grid.dy_face_h,
        grid.dArea_h,
        Nx, Ny,
        HGRID,
    )

    return nothing
end


function predictor_baroclinic!(
    prog::Prognostic,
    hist::History,
    intm::Intermediate,
    temp::Temporary,
    intp::Interpolated,
    forc::Forcing,
    diag::Diagnostic,
    grid::Grid,
    params::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
    step::Int,
)
    dt = FT(params.dt)

    # Prognostic aliases at time n
    m = prog.m
    n = prog.n
    h = prog.h

    m_star = intm.m_star
    n_star = intm.n_star
    h_star = intm.h_star

    rhs_m_tm0 = hist.rhs_m_tm0
    rhs_n_tm0 = hist.rhs_n_tm0
    rhs_h_tm0 = hist.rhs_h_tm0

    rhs_m_tm1 = hist.rhs_m_tm1
    rhs_n_tm1 = hist.rhs_n_tm1
    rhs_h_tm1 = hist.rhs_h_tm1

    rhs_m_tm2 = hist.rhs_m_tm2
    rhs_n_tm2 = hist.rhs_n_tm2
    rhs_h_tm2 = hist.rhs_h_tm2

    # Compute RHS at time n
    compute_baroclinic_rhs!(
        rhs_m_tm0,
        rhs_n_tm0,
        rhs_h_tm0,
        m, n, h,
        diag, temp, intp,
        grid, params,
        threads1, blocks1,
        threads2, blocks2,
    )

    # Bootstrap AB3 history on first baroclinic step
    if step == 1
        @. rhs_m_tm1 = rhs_m_tm0
        @. rhs_n_tm1 = rhs_n_tm0
        @. rhs_h_tm1 = rhs_h_tm0

        @. rhs_m_tm2 = rhs_m_tm0
        @. rhs_n_tm2 = rhs_n_tm0
        @. rhs_h_tm2 = rhs_h_tm0
    end

    # AB3 coefficients
    c0 = FT(23) / FT(12)
    c1 = -FT(16) / FT(12)
    c2 =  FT(5)  / FT(12)

    # AB3 update to get q^{n+1,pre}
    @. m_star = m + dt * (c0 * rhs_m_tm0 + c1 * rhs_m_tm1 + c2 * rhs_m_tm2)
    @. n_star = n + dt * (c0 * rhs_n_tm0 + c1 * rhs_n_tm1 + c2 * rhs_n_tm2)
    @. h_star = h + dt * (c0 * rhs_h_tm0 + c1 * rhs_h_tm1 + c2 * rhs_h_tm2)

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_u!(m_star, params.Nx, params.Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n_star, params.Nx, params.Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h_star, params.Nx, params.Ny)

    return nothing
end


function corrector_baroclinic!(
    prog::Prognostic,
    hist::History,
    intm::Intermediate,
    temp::Temporary,
    intp::Interpolated,
    forc::Forcing,
    diag::Diagnostic,
    grid::Grid,
    params::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
)
    dt = FT(params.dt)

    m = prog.m
    n = prog.n
    h = prog.h

    m_star = intm.m_star
    n_star = intm.n_star
    h_star = intm.h_star

    rhs_m_tp1 = hist.rhs_m_tp1
    rhs_n_tp1 = hist.rhs_n_tp1
    rhs_h_tp1 = hist.rhs_h_tp1

    rhs_m_tm0 = hist.rhs_m_tm0
    rhs_n_tm0 = hist.rhs_n_tm0
    rhs_h_tm0 = hist.rhs_h_tm0

    rhs_m_tm1 = hist.rhs_m_tm1
    rhs_n_tm1 = hist.rhs_n_tm1
    rhs_h_tm1 = hist.rhs_h_tm1

    rhs_m_tm2 = hist.rhs_m_tm2
    rhs_n_tm2 = hist.rhs_n_tm2
    rhs_h_tm2 = hist.rhs_h_tm2

    # Recompute RHS at q^{n+1,pre}
    compute_baroclinic_rhs!(
        rhs_m_tp1,
        rhs_n_tp1,
        rhs_h_tp1,
        m_star, n_star, h_star,
        diag, temp, intp,
        grid, params,
        threads1, blocks1,
        threads2, blocks2,
    )

    # AM3 coefficients
    c0 = FT(5)  / FT(12)
    c1 = FT(8)  / FT(12)
    c2 = -FT(1) / FT(12)

    # Temporary storage for q^{n+1}
    m_tp1 = temp.temp_var_x2
    n_tp1 = temp.temp_var_y2
    h_tp1 = temp.temp_var_x3

    # AM3 update to get q^{n+1}
    @. m_tp1 = m + dt * (c0 * rhs_m_tp1 + c1 * rhs_m_tm0 + c2 * rhs_m_tm1)
    @. n_tp1 = n + dt * (c0 * rhs_n_tp1 + c1 * rhs_n_tm0 + c2 * rhs_n_tm1)
    @. h_tp1 = h + dt * (c0 * rhs_h_tp1 + c1 * rhs_h_tm0 + c2 * rhs_h_tm1)

    # Shapiro filter on h
    @cuda threads=threads2 blocks=blocks2 k_apply_shapiro_filter!(
        h,
        h_tp1,
        params.smoothing_eps,
        params.Nx, params.Ny,
    )

    # Update m, n to time n+1
    @. m = m_tp1
    @. n = n_tp1

    # Rotate histories: (tm2 <- tm1), (tm1 <- tm0), (tm0 <- tp1)
    @. rhs_m_tm2 = rhs_m_tm1
    @. rhs_n_tm2 = rhs_n_tm1
    @. rhs_h_tm2 = rhs_h_tm1

    @. rhs_m_tm1 = rhs_m_tm0
    @. rhs_n_tm1 = rhs_n_tm0
    @. rhs_h_tm1 = rhs_h_tm0

    @. rhs_m_tm0 = rhs_m_tp1
    @. rhs_n_tm0 = rhs_n_tp1
    @. rhs_h_tm0 = rhs_h_tp1

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_u!(m, params.Nx, params.Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n, params.Nx, params.Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h, params.Nx, params.Ny)

    return nothing
end


function step_baroclinic!(
    state::State,
    grid::Grid,
    params::Params;
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
    step::Int,
)
    prog1 = state.prog1
    diag1 = state.diag1
    hist1 = state.hist1
    intm1 = state.intm1

    prog2 = state.prog2
    diag2 = state.diag2
    hist2 = state.hist2
    intm2 = state.intm2

    temp = state.temp
    intp = state.intp
    forc = state.forc

    Nx = Int(params.Nx)
    Ny = Int(params.Ny)

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(prog1.h, params.Nx, params.Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(prog2.h, params.Nx, params.Ny)

    calc_pressure_all!(
        diag1.pressure,
        diag2.pressure,
        prog1.h, prog2.h,
        params.g, params.rho1,
        params.rho2,
        threads1, blocks1,
        Nx, Ny,
    )

    # Predictor (AB3): layer 1 & 2
    predictor_baroclinic!(
        prog1, hist1, intm1,
        temp, intp, forc,
        diag1, grid, params,
        threads1, blocks1,
        threads2, blocks2,
        step,
    )

    predictor_baroclinic!(
        prog2, hist2, intm2,
        temp, intp, forc,
        diag2, grid, params,
        threads1, blocks1,
        threads2, blocks2,
        step,
    )

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(intm1.h_star, params.Nx, params.Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(intm2.h_star, params.Nx, params.Ny)

    calc_pressure_all!(
        diag1.pressure,
        diag2.pressure,
        intm1.h_star, intm2.h_star,
        params.g, params.rho1,
        params.rho2,
        threads1, blocks1,
        Nx, Ny,
    )

    # Corrector (AM3): layer 1 & 2
    corrector_baroclinic!(
        prog1, hist1, intm1,
        temp, intp, forc,
        diag1, grid, params,
        threads1, blocks1,
        threads2, blocks2,
    )

    corrector_baroclinic!(
        prog2, hist2, intm2,
        temp, intp, forc,
        diag2, grid, params,
        threads1, blocks1,
        threads2, blocks2,
    )

    return nothing
end