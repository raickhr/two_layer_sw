# forcing.jl
#
# Wind + stress forcing utilities for the 2-layer rotating shallow-water model.
#
# Provides:
#   - build_analytical_wind!(state, p)
#   - k_calc_surface_windStress!      (GPU kernel)
#   - calc_surface_windStress!        (wrapper; generic)
#   - k_calc_bottom_drag!             (GPU kernel)
#   - calc_bottom_drag!               (wrapper; generic)
#   - update_surface_and_bottom_stress!(state, forcing, grid, p; threads2, blocks2)
#
# Assumptions:
#   - FT, State, Forcing, Params, iper, clamp1 are defined in the module.
#   - Params has fields: Nx, Ny, lat1, lat2, dlat, u10, rho_air, Cd1, rho2, Cd2, hmin
#   - `state.prog` contains: h1, h2, H, m1, m2, M, n1, n2, N
#   - `state.forc` contains: uwind, vwind, taux_sf, tauy_sf, taux_bt, tauy_bt
#   - `state.temp` provides CuArray{FT,2} scratch (see stateVars.jl)
#   - CUDA and reconstruct.jl kernels are available:
#         k_recon_h_in_u!, k_recon_h_in_v!, k_recon_u_in_v!, k_recon_v_in_u!


# ----------------------------------------
# Analytical 10m wind (CPU -> GPU copy)
# ----------------------------------------

"""
    build_analytical_wind!(state, p)

Build an analytical 10m wind field with a sinusoidal meridional profile:

    u10(y) = p.u10 * sin(π * (lat - lat1) / (lat2 - lat1))
    v10(y) = 0

The result is stored in:
    state.forc.uwind, state.forc.vwind  (CuArray{FT,2})
"""
function build_analytical_wind!(state::State, p::Params)
    Nx, Ny = p.Nx, p.Ny

    # CPU arrays
    u10_cpu = Array{FT}(undef, Nx, Ny)
    v10_cpu = Array{FT}(undef, Nx, Ny)

    ΔLAT    = FT(p.lat2 - p.lat1)
    invΔLAT = FT(1) / ΔLAT

    @inbounds for j in 1:Ny
        δLAT = FT(j - 1) * FT(p.dlat)
        uval = FT(p.u10) * sin(FT(pi) * δLAT * invΔLAT)
        u10_cpu[:, j] .= uval
    end

    fill!(v10_cpu, FT(0))

    # Copy into preallocated GPU arrays
    copyto!(state.forc.uwind, u10_cpu)
    copyto!(state.forc.vwind, v10_cpu)

    return nothing
end


# ----------------------------------------
# Surface wind stress: relative wind formula
# ----------------------------------------

"""
GPU kernel: compute surface wind stress from relative wind:

    u_rel = u10 - u_ocean
    v_rel = v10 - v_ocean
    |U_rel| = sqrt(u_rel^2 + v_rel^2)

    τx = ρ_air * C_D * |U_rel| * u_rel
    τy = ρ_air * C_D * |U_rel| * v_rel

This kernel is agnostic to where the fields live (h/u/v); it just assumes
all inputs share the same staggering.
"""
function k_calc_surface_windStress!(
    taux_sf, tauy_sf,
    u10_2d, v10_2d,
    uo, vo,
    rho_air::FT,
    Cd1::FT,
    Nx::Int, Ny::Int
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        urel = u10_2d[i, j] - uo[i, j]
        vrel = v10_2d[i, j] - vo[i, j]

        relmag = sqrt(urel * urel + vrel * vrel)

        taux_sf[i, j] = rho_air * Cd1 * relmag * urel
        tauy_sf[i, j] = rho_air * Cd1 * relmag * vrel
    end

    return
end


"""
    calc_surface_windStress!(taux_out, tauy_out,
                             u10_2d, v10_2d,
                             uo, vo,
                             p, threads, blocks)

Generic wrapper for `k_calc_surface_windStress!`, not tied to `state.forc`.
"""
function calc_surface_windStress!(
    taux_out::CuArray{FT,2},
    tauy_out::CuArray{FT,2},
    u10_2d::CuArray{FT,2},
    v10_2d::CuArray{FT,2},
    uo::CuArray{FT,2},
    vo::CuArray{FT,2},
    p::Params,
    threads,
    blocks,
)
    Nx, Ny = p.Nx, p.Ny

    @cuda threads=threads blocks=blocks k_calc_surface_windStress!(
        taux_out,
        tauy_out,
        u10_2d,
        v10_2d,
        uo, vo,
        FT(p.rho_air),
        FT(p.Cd1),
        Nx, Ny
    )

    return nothing
end


# ----------------------------------------
# Bottom drag stress: quadratic drag
# ----------------------------------------

"""
GPU kernel: compute bottom drag stresses (quadratic drag):

Given bottom velocity (u, v):

    speed = sqrt(u^2 + v^2)
    τ = -ρ₂ * C_D2 * speed * (u, v)

So:
    τx = -ρ₂ * C_D2 * speed * u
    τy = -ρ₂ * C_D2 * speed * v
"""
function k_calc_bottom_drag!(
    taux_bt, tauy_bt,
    uo, vo,
    rho2::FT,
    Cd2::FT,
    Nx::Int, Ny::Int
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        u = uo[i, j]
        v = vo[i, j]

        speed = sqrt(u * u + v * v)
        c = -rho2 * Cd2 * speed

        taux_bt[i, j] = c * u
        tauy_bt[i, j] = c * v
    end

    return
end


"""
    calc_bottom_drag!(taux_out, tauy_out,
                      uo, vo,
                      p, threads, blocks)

Generic wrapper for `k_calc_bottom_drag!`.
"""
function calc_bottom_drag!(
    taux_out::CuArray{FT,2},
    tauy_out::CuArray{FT,2},
    uo::CuArray{FT,2},
    vo::CuArray{FT,2},
    p::Params,
    threads,
    blocks,
)
    Nx, Ny = p.Nx, p.Ny

    @cuda threads=threads blocks=blocks k_calc_bottom_drag!(
        taux_out,
        tauy_out,
        uo, vo,
        FT(p.rho2),
        FT(p.Cd2),
        Nx, Ny
    )

    return nothing
end


# ----------------------------------------
# High-level forcing wrapper
# ----------------------------------------

"""
    update_surface_and_bottom_stress!(
        state, forcing, grid, p;
        threads2, blocks2
    )

High-level wrapper used in the main driver:

    update_surface_and_bottom_stress!(
        state, state.forc, grid, p;
        threads2=threads2, blocks2=blocks2
    )

It computes stresses on the proper C-grid locations:

- τx on u-points (same as M, m₁, m₂)
- τy on v-points (same as N, n₁, n₂)

Algorithm:

1. Reconstruct thickness at u/v for each layer and total:
       h1_in_u, h1_in_v, h2_in_u, h2_in_v, H_in_u, H_in_v

2. Build native-grid velocities from transports:
       u1_u = m1 / h1_in_u,   u2_u = m2 / h2_in_u,   U_u = M / H_in_u
       v1_v = n1 / h1_in_v,   v2_v = n2 / h2_in_v,   V_v = N / H_in_v

   Then full velocities:
       u_surf_u = u1_u + U_u
       v_surf_v = v1_v + V_v
       u_bot_u  = u2_u + U_u
       v_bot_v  = v2_v + V_v

3. Reconstruct surface & bottom velocities across grids:
       v_surf_u = v_surf on u-grid    (k_recon_v_in_u!)
       u_surf_v = u_surf on v-grid    (k_recon_u_in_v!)
       v_bot_u  = v_bot  on u-grid
       u_bot_v  = u_bot  on v-grid

4. Reconstruct winds from h → u and h → v:
       (uwind_u, vwind_u) from (uwind, vwind) via k_recon_h_in_u!
       (uwind_v, vwind_v) via k_recon_h_in_v!

5. Call stress kernels:

   - Surface:
       τx(u) = k_calc_surface_windStress!(taux_sf, scratch, uwind_u, vwind_u, u_surf_u, v_surf_u, ...)
       τy(v) = k_calc_surface_windStress!(scratch, tauy_sf, uwind_v, vwind_v, u_surf_v, v_surf_v, ...)

   - Bottom:
       τx(u) = k_calc_bottom_drag!(taux_bt, scratch, u_bot_u, v_bot_u, ...)
       τy(v) = k_calc_bottom_drag!(scratch, tauy_bt, u_bot_v, v_bot_v, ...)
"""
function update_surface_and_bottom_stress!(
    state::State,
    p::Params;
    threads2,
    blocks2
)
    Nx, Ny = p.Nx, p.Ny
    prog   = state.prog
    temp   = state.temp

    hmin = FT(p.hmin)

    # --------------------------------------------------------
    # 1. Reconstruct thicknesses at u- and v-points
    # --------------------------------------------------------
    h1_in_u = temp.temp_var_x1
    h1_in_v = temp.temp_var_y1
    h2_in_u = temp.temp_var_x2
    h2_in_v = temp.temp_var_y2
    H_in_u  = temp.temp_var_x3
    H_in_v  = temp.temp_var_y3

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h1_in_u, prog.h1, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h1_in_v, prog.h1, Nx, Ny, hmin)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h2_in_u, prog.h2, Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h2_in_v, prog.h2, Nx, Ny, hmin)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(H_in_u,  prog.H,  Nx, Ny, hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(H_in_v,  prog.H,  Nx, Ny, hmin)

    # --------------------------------------------------------
    # 2. Native-grid velocities from transports
    # --------------------------------------------------------
    # u-grid velocities
    u1_u = temp.temp_var_x4
    u2_u = temp.temp_var_x5
    U_u  = temp.temp_var_x6

    @. u1_u = prog.m1 / h1_in_u
    @. u2_u = prog.m2 / h2_in_u
    @. U_u  = prog.M  / H_in_u

    # v-grid velocities
    v1_v = temp.temp_var_y4
    v2_v = temp.temp_var_y5
    V_v  = temp.temp_var_y6

    @. v1_v = prog.n1 / h1_in_v
    @. v2_v = prog.n2 / h2_in_v
    @. V_v  = prog.N  / H_in_v

    # Full-layer velocities (on native grids)
    u_surf_u = u1_u   # overwrite in-place
    v_surf_v = v1_v
    u_bot_u  = u2_u
    v_bot_v  = v2_v

    @. u_surf_u = u1_u + U_u
    @. v_surf_v = v1_v + V_v
    @. u_bot_u  = u2_u + U_u
    @. v_bot_v  = v2_v + V_v

    # --------------------------------------------------------
    # 3. Cross-grid reconstructions of velocities
    # --------------------------------------------------------
    v_surf_u = temp.temp_var_y7   # v at u-points
    u_surf_v = temp.temp_var_x7   # u at v-points
    v_bot_u  = temp.temp_var_y8   # v at u-points (bottom)
    u_bot_v  = temp.temp_var_x8   # u at v-points (bottom)

    @cuda threads=threads2 blocks=blocks2 k_recon_v_in_u!(v_surf_u, v_surf_v, Nx, Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_u_in_v!(u_surf_v, u_surf_u, Nx, Ny)

    @cuda threads=threads2 blocks=blocks2 k_recon_v_in_u!(v_bot_u,  v_bot_v,  Nx, Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_u_in_v!(u_bot_v,  u_bot_u,  Nx, Ny)

    # --------------------------------------------------------
    # 4. Winds reconstructed from h → u and h → v
    # --------------------------------------------------------
    uwind = state.forc.uwind
    vwind = state.forc.vwind

    uwind_u = temp.temp_var_x9
    vwind_u = temp.temp_var_y9
    uwind_v = h2_in_u   # reuse (no longer needed)
    vwind_v = h2_in_v   # reuse

    zero_hmin = FT(0)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(uwind_u, uwind, Nx, Ny, zero_hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(vwind_u, vwind, Nx, Ny, zero_hmin)

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(uwind_v, uwind, Nx, Ny, zero_hmin)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(vwind_v, vwind, Nx, Ny, zero_hmin)

    # --------------------------------------------------------
    # 5. Surface stresses: τx on u-grid, τy on v-grid
    # --------------------------------------------------------
    rho_air = FT(p.rho_air)
    Cd1     = FT(p.Cd1)

    taux_sf = state.forc.taux_sf
    tauy_sf = state.forc.tauy_sf
    tau_tmp = H_in_u                 # scratch (H_in_u no longer needed)

    # τx on u-points
    @cuda threads=threads2 blocks=blocks2 k_calc_surface_windStress!(
        taux_sf, tau_tmp,
        uwind_u, vwind_u,
        u_surf_u, v_surf_u,
        rho_air, Cd1,
        Nx, Ny
    )

    # τy on v-points
    @cuda threads=threads2 blocks=blocks2 k_calc_surface_windStress!(
        tau_tmp, tauy_sf,
        uwind_v, vwind_v,
        u_surf_v, v_surf_v,
        rho_air, Cd1,
        Nx, Ny
    )

    # --------------------------------------------------------
    # 6. Bottom drag stresses: τx on u-grid, τy on v-grid
    # --------------------------------------------------------
    rho2 = FT(p.rho2)
    Cd2  = FT(p.Cd2)

    taux_bt = state.forc.taux_bt
    tauy_bt = state.forc.tauy_bt
    tau_tmp2 = H_in_v               # second scratch (H_in_v no longer needed)

    # τx on u-points (bottom)
    @cuda threads=threads2 blocks=blocks2 k_calc_bottom_drag!(
        taux_bt, tau_tmp2,
        u_bot_u, v_bot_u,
        rho2, Cd2,
        Nx, Ny
    )

    # τy on v-points (bottom)
    @cuda threads=threads2 blocks=blocks2 k_calc_bottom_drag!(
        tau_tmp2, tauy_bt,
        u_bot_v, v_bot_v,
        rho2, Cd2,
        Nx, Ny
    )

    return nothing
end
