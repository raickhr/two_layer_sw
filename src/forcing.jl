############################################################
# forcing.jl
#
# Surface and bottom forcing for the rotating shallow-water
# C-grid model.
############################################################


# ---------------------------------------------------------
# Analytical 10 m wind (CPU → GPU)
# ---------------------------------------------------------

"""
build_analytical_wind!(state, p)

Build a sinusoidal meridional 10 m zonal wind profile

    u10(lat) = p.u10 * sin(π (lat - lat1) / (lat2 - lat1))
    v10(lat) = 0

and store it in state.forc.uwind (UGRID) and state.forc.vwind (VGRID).
"""
function build_analytical_wind!(state::State, p::Params)
    Nx, Ny = p.Nx, p.Ny

    u10_cpu = Array{FT}(undef, Nx, Ny)
    v10_cpu = Array{FT}(undef, Nx, Ny)

    ΔLAT    = FT(p.lat2 - p.lat1)
    invΔLAT = FT(1) / ΔLAT
    dlat    = FT(p.dlat)
    amp     = FT(p.u10)

    @inbounds for j in 1:Ny
        δLAT = FT(j - 2) * dlat
        uval = amp * sin(FT(pi) * δLAT * invΔLAT)
        u10_cpu[:, j] .= uval
    end

    fill!(v10_cpu, FT(0))

    copyto!(state.forc.uwind, u10_cpu)
    copyto!(state.forc.vwind, v10_cpu)

    return nothing
end


# ---------------------------------------------------------
# Surface and bottom stresses (layers 1 and 2)
# ---------------------------------------------------------

"""
update_surface_and_bottom_stress!(state, p; threads2, blocks2)

Compute:
  - layer-1 surface wind stress (taux_sf, tauy_sf),
  - layer-2 bottom drag stress (taux_bt, tauy_bt),

on UGRID (τx) and VGRID (τy) using quadratic drag laws.
"""
function update_surface_and_bottom_stress!(
    state::State,
    p::Params;
    threads2,
    blocks2,
)
    Nx, Ny = p.Nx, p.Ny

    prog1 = state.prog1
    prog2 = state.prog2
    diag1 = state.diag1
    diag2 = state.diag2
    intp  = state.intp
    temp  = state.temp

    hmin = FT(p.hmin)

    rho_air   = FT(p.rho_air)
    Cd1       = FT(p.Cd1)
    rho_water = FT(p.rho2)
    Cd2       = FT(p.Cd2)

    h_in_u = intp.h_in_u
    h_in_v = intp.h_in_v
    u_in_v = intp.u_in_v
    v_in_u = intp.v_in_u

    # =========================
    # Layer 1: surface stress
    # =========================

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h_in_u, prog1.h, Nx, Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h_in_v, prog1.h, Nx, Ny)

    @. h_in_u = max(h_in_u, hmin)
    @. h_in_v = max(h_in_v, hmin)

    u = diag1.u
    v = diag1.v

    @. u = prog1.m / h_in_u
    @. v = prog1.n / h_in_v

    @cuda threads=threads2 blocks=blocks2 k_recon_v_in_u!(v_in_u, v, Nx, Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_u_in_v!(u_in_v, u, Nx, Ny)

    uwind = state.forc.uwind
    vwind = state.forc.vwind

    uwind_in_v = temp.temp_var_x1
    vwind_in_u = temp.temp_var_y1

    @cuda threads=threads2 blocks=blocks2 k_recon_v_in_u!(vwind_in_u, vwind, Nx, Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_u_in_v!(uwind_in_v, uwind, Nx, Ny)

    taux_sf = state.forc.taux_sf
    tauy_sf = state.forc.tauy_sf

    rel_mag_in_u = temp.temp_var_x2
    rel_mag_in_v = temp.temp_var_y2

    @. rel_mag_in_u = sqrt((uwind - u)^2 + (vwind_in_u - v_in_u)^2)
    @. rel_mag_in_v = sqrt((uwind_in_v - u_in_v)^2 + (vwind - v)^2)

    @. taux_sf = rho_air * Cd1 * rel_mag_in_u * (uwind - u)
    @. tauy_sf = rho_air * Cd1 * rel_mag_in_v * (vwind - v)

    # =========================
    # Layer 2: bottom drag
    # =========================

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h_in_u, prog2.h, Nx, Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h_in_v, prog2.h, Nx, Ny)

    @. h_in_u = max(h_in_u, hmin)
    @. h_in_v = max(h_in_v, hmin)

    u = diag2.u
    v = diag2.v

    @. u = prog2.m / h_in_u
    @. v = prog2.n / h_in_v

    @cuda threads=threads2 blocks=blocks2 k_recon_v_in_u!(v_in_u, v, Nx, Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_u_in_v!(u_in_v, u, Nx, Ny)

    @. rel_mag_in_u = sqrt(u^2      + v_in_u^2)
    @. rel_mag_in_v = sqrt(u_in_v^2 + v^2)

    @. state.forc.taux_bt = -rho_water * Cd2 * rel_mag_in_u * u
    @. state.forc.tauy_bt = -rho_water * Cd2 * rel_mag_in_v * v

    return nothing
end