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
# one baroclinic time step params.dt:
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
# hist::history stores *tendency histories*:
#   rhs_*_tm0 ≈ R(q^n)
#   rhs_*_tm1 ≈ R(q^{n−1})
#   rhs_*_tm2 ≈ R(q^{n−2})
#   rhs_*_tp1 ≈ R(q^{n+1,pre})
#
# intm::intermediate holds the provisional AB3-predicted fields:
#   h1_star, h2_star
#   m1_star, m2_star
#   n1_star, n2_star
############################################################

function calc_pressure_all!(
    p1, p2, h1, h2,
    g::FT,
    gp::FT, 
    threads1::Int,
    blocks1::Int,
    Nx::Int, 
    Ny::Int)

    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h1,  Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h2,  Nx, Ny)

    @. p1 = g * (h1 + h2)
    @. p2 = g * (h1 + h2) + gp * h2
    return nothing
end


function compute_baroclinic_rhs!(
    rhs_m::CuArray{FT,2},
    rhs_n::CuArray{FT,2},
    rhs_h::CuArray{FT,2},
    m::CuArray{FT,2}, 
    n::CuArray{FT,2}, 
    h::CuArray{FT,2},
    pressure::CuArray{FT,2},
    ρ::FT,
    temp::Temporary,
    intp::interpolated,
    grid::Grid,
    params::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
)

    # Grid sizes as floating type
    Nx = Int(params.Nx)
    Ny = Int(params.Ny)

    # Minimum thickness
    hmin = FT(params.hmin)

    # ------------------------------------------------------------------
    # Select layer-specific interpolated thickness on u / v grids
    # ------------------------------------------------------------------
    h_in_u = intp.h1_in_u
    h_in_v = intp.h1_in_v

    # ------------------------------------------------------------------
    # Initialize RHS to zero before accumulation
    # ------------------------------------------------------------------
    @. rhs_m = 0
    @. rhs_n = 0
    @. rhs_h = 0

    # ------------------------------------------------------------------
    # Apply wall BCs to prognostic fields (in-place)
    # ------------------------------------------------------------------
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n,  Nx, Ny)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_h!(h,  Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h_in_u, h, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h_in_v, h, Nx, Ny, hmin)

    ##########################################################
    #                  MOMENTUM CONSERVATION
    ##########################################################

    # ========================================================
    # 1. Pressure Gradient
    # ========================================================
    press_gradx = temp.temp_var_x2
    press_grady = temp.temp_var_y2

    dx_n2n_h = grid.dx_n2n_h
    dy_n2n_h = grid.dy_n2n_h

    @cuda threads=threads2 blocks=blocks2 k_calc_gradient!(
        press_gradx, press_grady,
        pressure,
        dx_n2n_h, dy_n2n_h,
        Nx, Ny,
    )

    @. rhs_m = rhs_m - press_gradx * h_in_u
    @. rhs_n = rhs_n - press_grady * h_in_v

    #@info "max |press_gradx|" maximum(abs.(Array(rhs_m)))
    #@info "max |press_grady|" maximum(abs.(Array(rhs_n)))

    # ========================================================
    # 2. Biharmonic term
    # ========================================================
    # Velocities on u/v points
    u = temp.temp_var_x1
    v = temp.temp_var_y1
    @. u = m / h_in_u
    @. v = n / h_in_v

    biharmonic_m = temp.temp_var_x2
    biharmonic_n = temp.temp_var_y2

    buf1_x = temp.temp_var_x3
    buf1_y = temp.temp_var_y3

    buf2_x = temp.temp_var_x4
    buf2_y = temp.temp_var_y4

    ν = FT(params.nu)

    calculate_biharmonic_term!(biharmonic_m, biharmonic_n,
                               u, v, h,
                               buf1_x, buf1_y,
                               buf2_x, buf2_y,
                               grid, threads2, blocks2,
                               ν, Nx, Ny)

    # viscous_m = temp.temp_var_x2
    # viscous_n = temp.temp_var_y2

    # buf1_x = temp.temp_var_x3
    # buf1_y = temp.temp_var_y3

    # buf2_x = temp.temp_var_x4
    # buf2_y = temp.temp_var_y4

    # buf3_x = temp.temp_var_x5
    # buf3_y = temp.temp_var_y5

    # buf4_x = temp.temp_var_x6
    # buf4_y = temp.temp_var_y6

    # buf5_x = temp.temp_var_x7
    # buf5_y = temp.temp_var_y7

    # calculate_viscous_term!(
    #     viscous_m, viscous_n,
    #     u, v, h,
    #     buf1_x, buf1_y,
    #     buf2_x, buf2_y,
    #     buf3_x, buf3_y,
    #     buf4_x, buf4_y,
    #     buf5_x, buf5_y,
    #     grid, threads2, blocks2,
    #     Nx, Ny
    # )

    # ADD biharmonic to RHS, do not overwrite
    @. rhs_m = rhs_m + biharmonic_m
    @. rhs_n = rhs_n + biharmonic_n

    # ========================================================
    # 3. Curvature Terms
    # ========================================================
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h_in_u, h, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h_in_v, h, Nx, Ny, hmin)

    curv_x = temp.temp_var_x2
    curv_y = temp.temp_var_y2

    lat_u = grid.lat_u
    lat_v = grid.lat_v

    @cuda threads=threads2 blocks=blocks2 k_calc_curvature_terms!(
        curv_x, curv_y,
        m, n,
        h_in_u, h_in_v,
        lat_u, lat_v,
        Nx, Ny,
        FT(params.earthRadius),
    )

    @. rhs_m = rhs_m +  curv_x
    @. rhs_n = rhs_n +  curv_y

    @. temp.temp_var_x8 = curv_x
    @. temp.temp_var_y8 = curv_y

    # ========================================================
    # 4. Coriolis Terms
    # ========================================================
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h_in_u, h, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h_in_v, h, Nx, Ny, hmin)

    @. u = m / h_in_u
    @. v = n / h_in_v

    cor_x = temp.temp_var_x2
    cor_y = temp.temp_var_y2

    @. cor_x = 0.0
    @. cor_y = 0.0

    u_in_v = temp.temp_var_x3
    v_in_u = temp.temp_var_y3

    @cuda threads=threads2 blocks=blocks2 k_recon_u_in_v!(u_in_v, u, Nx, Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_v_in_u!(v_in_u, v, Nx, Ny)

    @. cor_x =  h_in_u * grid.f_u * v_in_u
    @. cor_y = -h_in_v * grid.f_v * u_in_v

    @. rhs_m = rhs_m + cor_x
    @. rhs_n = rhs_n + cor_y

    #@info "predictor max |cor_x| " maximum(abs.(Array(cor_x)))
    #@info "predictor max |cor_y| " maximum(abs.(Array(cor_y)))

    # ========================================================
    # 5. Advection Terms
    # ========================================================
    adv_m = temp.temp_var_x2
    adv_n = temp.temp_var_y2

    # u_face_for_u = temp.temp_var_x3
    # v_face_for_u = temp.temp_var_y3

    # u_face_for_v = temp.temp_var_x4
    # v_face_for_v = temp.temp_var_y4

    # # # u-cells (advect m)
    # @cuda threads=threads2 blocks=blocks2 k_calc_faceVels_for_ucell!(
    #     u_face_for_u, v_face_for_u,
    #     m, n,
    #     h,
    #     Nx, Ny,
    #     hmin,
    # )
    # @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(v_face_for_u, Nx, Ny)

    # dx_face_u = grid.dx_face_u
    # dy_face_u = grid.dy_face_u
    # dArea_u   = grid.dArea_u

    # @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
    #     adv_m,
    #     m,
    #     u_face_for_u, v_face_for_u,
    #     dx_face_u, dy_face_u,
    #     dArea_u,
    #     Nx, Ny,
    #     UGRID,
    # )

    @cuda threads=threads2 blocks=blocks2 k_calc_flux_m_for_ucell!(adv_m,               
                                                                    m, n,                 
                                                                    h,  
                                                                    grid.dx_face_u, grid.dy_face_u, # face lengths around UGRID cells
                                                                    grid.dArea_u,              # u-cell area                       
                                                                    params.Nx, params.Ny,
                                                                    params.hmin)

    @. rhs_m = rhs_m + adv_m

    # v-cells (advect n)
    # @cuda threads=threads2 blocks=blocks2 k_calc_faceVels_for_vcell!(
    #     u_face_for_v, v_face_for_v,
    #     m, n,
    #     h,
    #     Nx, Ny,
    #     hmin,
    # )
    # @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(v_face_for_v, Nx, Ny)

    # dx_face_v = grid.dx_face_v
    # dy_face_v = grid.dy_face_v
    # dArea_v   = grid.dArea_v

    # @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
    #     adv_n,
    #     n,
    #     u_face_for_v, v_face_for_v,
    #     dx_face_v, dy_face_v,
    #     dArea_v,
    #     Nx, Ny,
    #     VGRID,
    # )

    @cuda threads=threads2 blocks=blocks2 k_calc_flux_n_for_vcell!(adv_n,               
                                                                    m, n,                 
                                                                    h, 
                                                                    grid.dx_face_v, grid.dy_face_v, # face lengths around VGRID cells
                                                                    grid.dArea_v,              # v-cell area                   
                                                                    params.Nx, params.Ny,
                                                                    params.hmin)

    @. rhs_n = rhs_n + adv_n

    ##########################################################
    #                  MASS CONSERVATION
    ##########################################################
    @. u = m / h_in_u
    @. v = n / h_in_v
    
    dx_face_h = grid.dx_face_h
    dy_face_h = grid.dy_face_h
    dArea_h   = grid.dArea_h

    @cuda threads=threads2 blocks=blocks2 k_calc_WENOZ_flux2d!(
        rhs_h,
        h,
        u, v,
        dx_face_h, dy_face_h,
        dArea_h,
        Nx, Ny,
        HGRID,
    )

    return nothing
end


function predictor_baroclinic!(
    prog::Prognostic,
    hist::history,
    intm::intermediate,
    temp::Temporary,
    intp::interpolated,
    forc::Forcing,
    grid::Grid,
    params::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
    step::Int,
    layer::Int,
)

    dt = FT(params.dt)

    # Prognostic aliases at time n
    if layer == 1
        m = prog.m1
        n = prog.n1
        h = prog.h1

        m_star = intm.m1_star
        n_star = intm.n1_star
        h_star = intm.h1_star

        pressure = temp.pressure1
        ρ = params.rho1

        rhs_m_tm0 = hist.rhs_m1_tm0
        rhs_n_tm0 = hist.rhs_n1_tm0
        rhs_h_tm0 = hist.rhs_h1_tm0

        rhs_m_tm1 = hist.rhs_m1_tm1
        rhs_n_tm1 = hist.rhs_n1_tm1
        rhs_h_tm1 = hist.rhs_h1_tm1

        rhs_m_tm2 = hist.rhs_m1_tm2
        rhs_n_tm2 = hist.rhs_n1_tm2
        rhs_h_tm2 = hist.rhs_h1_tm2
    else
        m = prog.m2
        n = prog.n2
        h = prog.h2

        m_star = intm.m2_star
        n_star = intm.n2_star
        h_star = intm.h2_star

        pressure = temp.pressure2
        ρ = params.rho2

        rhs_m_tm0 = hist.rhs_m2_tm0
        rhs_n_tm0 = hist.rhs_n2_tm0
        rhs_h_tm0 = hist.rhs_h2_tm0

        rhs_m_tm1 = hist.rhs_m2_tm1
        rhs_n_tm1 = hist.rhs_n2_tm1
        rhs_h_tm1 = hist.rhs_h2_tm1

        rhs_m_tm2 = hist.rhs_m2_tm2
        rhs_n_tm2 = hist.rhs_n2_tm2
        rhs_h_tm2 = hist.rhs_h2_tm2
    end

    compute_baroclinic_rhs!(
        rhs_m_tm0,
        rhs_n_tm0,
        rhs_h_tm0,
        m, n, h,
        pressure, ρ,
        temp, intp,
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

    #@info "predictor max |rhs_m| layer=$layer" maximum(abs.(Array(rhs_h_tm0)))
    #@info "predictor max |rhs_h| layer=$layer" maximum(abs.(Array(rhs_h_tm0)))

    # AB3 coefficients
    c0 = FT(23) / FT(12)
    c1 = -FT(16) / FT(12)
    c2 =  FT(5)  / FT(12)

    # @. m_star = m + dt * rhs_m_tm0 
    # @. n_star = n + dt * rhs_n_tm0 
    # @. h_star = h + dt * rhs_h_tm0 

    # --------------------------------------------------------
    # AB3 update to get q^{n+1,pre}
    # --------------------------------------------------------
    @. m_star = m + dt * (c0 * rhs_m_tm0 + c1 * rhs_m_tm1 + c2 * rhs_m_tm2)
    @. n_star = n + dt * (c0 * rhs_n_tm0 + c1 * rhs_n_tm1 + c2 * rhs_n_tm2)
    @. h_star = h + dt * (c0 * rhs_h_tm0 + c1 * rhs_h_tm1 + c2 * rhs_h_tm2)

    return nothing
end


function corrector_baroclinic!(
    prog::Prognostic,
    hist::history,
    intm::intermediate,
    temp::Temporary,
    intp::interpolated,
    forc::Forcing,
    grid::Grid,
    params::Params,
    threads1::Int,
    blocks1::Int,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
    layer::Int,
)

    dt = FT(params.dt)

    if layer == 1
        m = prog.m1
        n = prog.n1
        h = prog.h1

        m_star = intm.m1_star
        n_star = intm.n1_star
        h_star = intm.h1_star
        pressure = temp.pressure1
        ρ = params.rho1

        rhs_m_tp1 = hist.rhs_m1_tp1
        rhs_n_tp1 = hist.rhs_n1_tp1
        rhs_h_tp1 = hist.rhs_h1_tp1

        rhs_m_tm0 = hist.rhs_m1_tm0
        rhs_n_tm0 = hist.rhs_n1_tm0
        rhs_h_tm0 = hist.rhs_h1_tm0

        rhs_m_tm1 = hist.rhs_m1_tm1
        rhs_n_tm1 = hist.rhs_n1_tm1
        rhs_h_tm1 = hist.rhs_h1_tm1

        rhs_m_tm2 = hist.rhs_m1_tm2
        rhs_n_tm2 = hist.rhs_n1_tm2
        rhs_h_tm2 = hist.rhs_h1_tm2
    else
        m = prog.m2
        n = prog.n2
        h = prog.h2

        m_star = intm.m2_star
        n_star = intm.n2_star
        h_star = intm.h2_star
        pressure = temp.pressure2
        ρ = params.rho2

        rhs_m_tp1 = hist.rhs_m2_tp1
        rhs_n_tp1 = hist.rhs_n2_tp1
        rhs_h_tp1 = hist.rhs_h2_tp1

        rhs_m_tm0 = hist.rhs_m2_tm0
        rhs_n_tm0 = hist.rhs_n2_tm0
        rhs_h_tm0 = hist.rhs_h2_tm0

        rhs_m_tm1 = hist.rhs_m2_tm1
        rhs_n_tm1 = hist.rhs_n2_tm1
        rhs_h_tm1 = hist.rhs_h2_tm1

        rhs_m_tm2 = hist.rhs_m2_tm2
        rhs_n_tm2 = hist.rhs_n2_tm2
        rhs_h_tm2 = hist.rhs_h2_tm2
    end

    # Recompute RHS at q^{n+1,pre}
    compute_baroclinic_rhs!(
        rhs_m_tp1,
        rhs_n_tp1,
        rhs_h_tp1,
        m_star, n_star, h_star,
        pressure, ρ,
        temp, intp,
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

    #@info "max |rhs_h*| layer=$layer" maximum(abs.(Array(rhs_h_tm0)))
    #@info "max |rhs_h| layer=$layer" maximum(abs.(Array(rhs_h_tp1)))

    # @. m_tp1 = m + FT(0.5) * dt * (rhs_m_tp1 +  rhs_m_tm0)
    # @. n_tp1 = n + FT(0.5) * dt * (rhs_n_tp1 +  rhs_n_tm0)
    # @. h_tp1 = h + FT(0.5) * dt * (rhs_h_tp1 +  rhs_h_tm0)

    #@info "corrector max |rhs_m| layer=$layer" maximum(abs.(Array(rhs_m_tp1)))
    #@info "corrector max |rhs_n| layer=$layer" maximum(abs.(Array(rhs_n_tp1)))

    # --------------------------------------------------------
    # AM3 update to get q^{n+1}
    # --------------------------------------------------------
    @. m_tp1 = m + dt * (c0 * rhs_m_tp1 + c1 * rhs_m_tm0 + c2 * rhs_m_tm1)
    @. n_tp1 = n + dt * (c0 * rhs_n_tp1 + c1 * rhs_n_tm0 + c2 * rhs_n_tm1)
    @. h_tp1 = h + dt * (c0 * rhs_h_tp1 + c1 * rhs_h_tm0 + c2 * rhs_h_tm1)


    # -------------------------
    # Shapiro filter on h
    # -------------------------
    @cuda threads=threads2 blocks=blocks2 k_apply_shapiro_filter!(
        h,
        h_tp1,
        params.smoothing_eps,
        params.Nx, params.Ny,
    )

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
    mode_split::Bool = true,
)
    prog = state.prog
    intp = state.intp
    intm = state.intm
    hist = state.hist
    temp = state.temp
    forc = state.forc

    @. prog.H = prog.h1 + prog.h2

    pressure1 = temp.pressure1
    pressure2 = temp.pressure2

    # Pressure at time n (from h1, h2)
    calc_pressure_all!(
        pressure1, pressure2,
        prog.h1, prog.h2,
        params.g, params.gp,
        threads1, blocks1,
        params.Nx, params.Ny
    )

    # Predictor (AB3): layer 1 & 2
    predictor_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, params,
        threads1, blocks1,
        threads2, blocks2,
        step , 1
    )

    predictor_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, params,
        threads1, blocks1,
        threads2, blocks2,
        step , 2,
    )

    # Pressure at time n+1,pre from star thicknesses
    pressure1 = temp.pressure1
    pressure2 = temp.pressure2

    calc_pressure_all!(
        pressure1, pressure2,
        intm.h1_star, intm.h2_star,
        params.g, params.gp,
        threads1, blocks1,
        params.Nx, params.Ny
    )

    # Corrector (AM3): layer 1 & 2
    corrector_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, params,
        threads1, blocks1,
        threads2, blocks2, 1,
    )

    corrector_baroclinic!(
        prog, hist, intm,
        temp, intp, forc,
        grid, params,
        threads1, blocks1,
        threads2, blocks2, 2,
    )

    return nothing
end