# initial_condition.jl
#
# Initial condition utilities for the 2-layer rotating shallow-water model.


# ============================================================
# 1. Gaussian surface "bulb" SSH anomaly (top layer only)
# ============================================================

"""
    build_h1_for_surface_bulb_eta(p; η0=1.0, R=1000e3, lon0=0.0, lat0=0.0)

Return a CPU array `h1(x,y) = p.H1 + η(x,y)` where η is a Gaussian
**surface** height anomaly (SSH) of amplitude `η0` and radius `R` (meters),
centered at (`lon0`, `lat0`) in degrees. The domain-mean of η is removed
so the perturbation is mass-conserving.
"""
function build_h1_for_surface_bulb_eta(
    p::Params;
    η0::FT   = FT(1.0),
    R::FT    = FT(1000e3),
    lon0::FT = FT(0.0),
    lat0::FT = FT(0.0),
)
    Nx, Ny = p.Nx, p.Ny

    deg2rad = FT(π) / FT(180)
    earthR  = FT(p.earthRadius)

    # 1D coordinates at h-points
    lon_h = FT(p.lon1) .+ FT.(0:Nx-1) .* FT(p.dlon)
    lat_h = FT(p.lat1) .+ FT.(0:Ny-1) .* FT(p.dlat)

    # 2D mesh
    lon_h2d = repeat(lon_h, 1, Ny)
    lat_h2d = repeat(lat_h', Nx, 1)

    δlon = lon_h2d .- lon0
    δlat = lat_h2d .- lat0

    # Tangent-plane distances about (lon0, lat0)
    cosφ0 = cos(lat0 * deg2rad)
    dx    = earthR * cosφ0 .* (δlon .* deg2rad)
    dy    = earthR .*        (δlat .* deg2rad)
    r2    = dx.^2 .+ dy.^2

    η = η0 .* exp.(-r2 ./ (R * R))
    η .-= mean(η)

    h1 = fill(FT(p.H1), Nx, Ny)
    h1 .+= η
    return h1
end


"""
    initialize_surface_bulb!(state, p; η0=1.0, R=1000e3, lon0=0.0, lat0=0.0)

Initialize the model with a **surface SSH bulb** in the top layer by
overwriting `state.prog.h1` and updating the total thickness fields
`H` and `H_old` via

    H = h1 + h2

(bottom fixed, free surface displaced).
"""
function initialize_surface_bulb!(
    state::State,
    p::Params;
    η0::FT   = FT(1.0),
    R::FT    = FT(1000e3),
    lon0::FT = FT(0.0),
    lat0::FT = FT(0.0),
)
    h1_cpu = build_h1_for_surface_bulb_eta(
        p; η0=η0, R=R, lon0=lon0, lat0=lat0
    )

    copyto!(state.prog.h1, h1_cpu)

    state.prog.H     .= state.prog.h1 .+ state.prog.h2
    state.prog.H_old .= state.prog.H

    return nothing
end


# ============================================================
# 2. Baroclinic tanh-jet: tilted interface ξ (internal mode)
# ============================================================

"""
    build_baroclinic_tanh_jet(
        p;
        Δξ=100.0,
        L=300e3,
        lat0=(p.lat1 + p.lat2)/2,
        noise_amp=0.05,
    )

Construct CPU fields `(h1, h2, u1, u2)` for a **baroclinic tanh jet**
obtained by tilting the internal interface ξ(y) while keeping the free
surface flat and total thickness constant.

Interface displacement:
    ξ(y) = Δξ * tanh(y / L)

Thicknesses (pure internal mode):
    h1 = H1 + ξ
    h2 = H2 - ξ
so that
    H = h1 + h2 = H1 + H2 (constant in y).

Thermal-wind shear (2-layer reduced-gravity):
    f0 (u1 − u2) = − gp * dξ/dy

where f0 is evaluated at latitude `lat0`.

A small random perturbation of amplitude `noise_amp * Δξ` is added to h1
(and subtracted from h2) to trigger baroclinic instability while
preserving H.
"""
function build_baroclinic_tanh_jet(
    p::Params;
    Δξ::FT        = FT(100.0),
    L::FT         = FT(300e3),
    lat0::FT      = FT(0.5 * (p.lat1 + p.lat2)),
    noise_amp::FT = FT(0.05),
)
    Nx, Ny = p.Nx, p.Ny
    H1, H2 = FT(p.H1), FT(p.H2)
    gp     = FT(p.gp)
    Ω      = FT(p.Ω)
    Rearth = FT(p.earthRadius)
    deg2rad = FT(π) / FT(180)

    # 1D y-coordinate about lat0
    lat_h = FT(p.lat1) .+ FT.(0:Ny-1) .* FT(p.dlat)
    y     = Rearth .* (lat_h .- lat0) .* deg2rad

    # Interface displacement ξ(y) and derivative
    ξ    = Δξ .* tanh.(y ./ L)
    sech2 = 1 ./ cosh.(y ./ L).^2
    dξdy = Δξ .* (sech2 ./ L)

    # Coriolis at lat0 and baroclinic shear
    f0 = 2 * Ω * sin(lat0 * deg2rad)

    ΔU = .-(gp / f0) .* dξdy          # u1 − u2 from thermal wind

    # Partition shear so that barotropic velocity is zero:
    # H1 u1 + H2 u2 = 0 => choose u1,u2 such that ⟨u⟩=0
    Htot  = H1 + H2
    u1_1d = ( H2 / Htot) .* ΔU
    u2_1d = ( H1 / Htot) .* (-ΔU)

    # Allocate fields
    h1 = Array{FT}(undef, Nx, Ny)
    h2 = Array{FT}(undef, Nx, Ny)
    u1 = Array{FT}(undef, Nx, Ny)
    u2 = Array{FT}(undef, Nx, Ny)

    @inbounds for j in 1:Ny
        h1[:, j] .= H1 + ξ[j]
        h2[:, j] .= H2 - ξ[j]
        u1[:, j] .= u1_1d[j]
        u2[:, j] .= u2_1d[j]
    end

    # Add small internal noise (preserve H)
    if noise_amp != 0
        noise = noise_amp * Δξ .* randn(FT, Nx, Ny)
        h1 .+= noise
        h2 .-= noise
    end

    return h1, h2, u1, u2
end


"""
    initialize_baroclinic_tanh_jet!(state, p;
        Δξ=100.0, L=300e3, lat0=(p.lat1 + p.lat2)/2, noise_amp=0.05)

Initialize the model with a **baroclinically unstable tanh jet** produced
by a tilted internal interface ξ(y):

- Sets `h1 = H1 + ξ`, `h2 = H2 − ξ` (constant total thickness).
- Sets zonal layer transports `m1 = h1 u1`, `m2 = h2 u2` in thermal-wind
  balance with ξ(y).
- Zeros meridional and barotropic transports (`n1`, `n2`, `M`, `N`).
- Updates `H` and `H_old` to remain consistent with h1, h2.
"""
function initialize_baroclinic_tanh_jet!(
    state::State,
    p::Params;
    Δξ::FT        = FT(100.0),
    L::FT         = FT(300e3),
    lat0::FT      = FT(0.5 * (p.lat1 + p.lat2)),
    noise_amp::FT = FT(0.05),
)
    h1_cpu, h2_cpu, u1_cpu, u2_cpu =
        build_baroclinic_tanh_jet(
            p; Δξ=Δξ, L=L, lat0=lat0, noise_amp=noise_amp
        )

    Nx, Ny = p.Nx, p.Ny

    # Copy layers
    copyto!(state.prog.h1, h1_cpu)
    copyto!(state.prog.h2, h2_cpu)

    # Total thickness (pure internal mode: constant in y)
    state.prog.H     .= state.prog.h1 .+ state.prog.h2
    state.prog.H_old .= state.prog.H

    # Zonal transports from layer velocities
    m1_cpu = Array{FT}(undef, Nx, Ny)
    m2_cpu = Array{FT}(undef, Nx, Ny)

    @inbounds for j in 1:Ny, i in 1:Nx
        m1_cpu[i, j] = h1_cpu[i, j] * u1_cpu[i, j]
        m2_cpu[i, j] = h2_cpu[i, j] * u2_cpu[i, j]
    end

    copyto!(state.prog.m1, m1_cpu)
    copyto!(state.prog.m2, m2_cpu)

    # Zero meridional and barotropic transports
    fill!(state.prog.n1, FT(0))
    fill!(state.prog.n2, FT(0))
    fill!(state.prog.M,  FT(0))
    fill!(state.prog.N,  FT(0))

    return nothing
end
