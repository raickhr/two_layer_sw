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

    ξ0 = FT(1027.0)/FT(2.0) * η0   
    ξ = ξ0 .* exp.(-r2 ./ (R * R))
    ξ .-= mean(ξ)

    h1 = fill(FT(p.H1), Nx, Ny)
    h2 = fill(FT(p.H2), Nx, Ny)

    @. h2 = h2 - ξ
    @. h1 = h1 + ξ + η    


    return h1, h2
end


"""
    initialize_surface_bulb!(state, p; η0=1.0, R=1000e3, lon0=0.0, lat0=0.0)

Initialize the model with a **surface SSH bulb** in the top layer by
overwriting `state.prog1.h` and updating the total thickness fields
`H` and `H_old` via

    H = h1 + h2

(bottom fixed, free surface displaced). Velocities / transports are
unchanged by this initializer.
"""
function initialize_surface_bulb!(
    state::State,
    p::Params;
    η0::FT   = FT(1.0),
    R::FT    = FT(1000e3),
    lon0::FT = FT(0.0),
    lat0::FT = FT(0.0),
)
    h1_cpu, h2_cpu = build_h1_for_surface_bulb_eta(
        p; η0=η0, R=R, lon0=lon0, lat0=lat0
    )

    copyto!(state.prog1.h, h1_cpu)
    copyto!(state.prog2.h, h2_cpu)

    #state.prog.H     .= state.prog1.h .+ state.prog2.h
    #state.prog.H_old .= state.prog.H

    return nothing
end


# ============================================================
# 2. Baroclinic tanh-jet: tilted interface ξ (internal mode)
# ============================================================

"""
    build_baroclinic_tanh_jet(p;
        Δξ = FT(5.0),
        L  = FT(300e3),
        lat0 = FT(0.5*(p.lat1+p.lat2)),
        noise_amp = FT(0.05),
    )

Construct a baroclinic tanh-jet state `(h1, h2, m1, m2)` for a 2-layer
reduced-gravity shallow-water model.

Interface displacement:
    ξ(y) = Δξ * tanh(y/L)

with total thickness preserved (pure internal thickness mode):
    h1 = H1 − ξ,     h2 = H2 + ξ,
so that H = h1 + h2 = H1 + H2 is constant.

Assuming surface height η = 0, the **zonal layer transports** are put
in geostrophic balance with ξ(y) via

    f m1 =  h1 g  ∂ξ/∂y
    f m2 = -h1 g  ∂ξ/∂y − h2 g' ∂ξ/∂y,

where f = f(lat) is the full latitude-dependent Coriolis parameter.
This generally yields a nonzero barotropic transport
    M = m1 + m2
(i.e. this is not a zero-barotropic internal mode; only H is purely
baroclinic).

Optionally, a small 2D baroclinic “noise” pattern of amplitude
~(noise_amp*Δξ) and horizontal scale 5·Ld is added to h1 (and subtracted
from h2) to seed instability while preserving H. Here Ld is the first
baroclinic deformation radius computed at f0 = f(lat0).

Returns: `(h1, h2, m1, m2)` as `(Nx, Ny)` Arrays.
"""
function build_baroclinic_tanh_jet(
    p::Params;
    Δξ::FT        = FT(10.0),
    L::FT         = FT(200e3),
    lat0::FT      = FT(0.5 * (p.lat1 + p.lat2)),
    noise_amp::FT = FT(0.1),
)
    Nx, Ny  = p.Nx, p.Ny
    H1, H2  = FT(p.H1), FT(p.H2)
    g       = FT(p.g)
    gp      = FT(p.gp)
    Ω       = FT(p.Ω)
    Rearth  = FT(p.earthRadius)
    deg2rad = FT(π) / FT(180)

    # ---------------------------------------------------------
    # 1) Latitude (h-points) and meridional coordinate y about lat0
    # ---------------------------------------------------------
    lat_h = FT(p.lat1) .+ FT.(0:Ny-1) .* FT(p.dlat)
    y     = Rearth .* (lat_h .- lat0) .* deg2rad    # size Ny

    h1_1d = fill(H1, Ny)
    h2_1d = fill(H2, Ny)

    m1_1d = Array{FT}(undef, Ny)
    m2_1d = Array{FT}(undef, Ny)

    # ---------------------------------------------------------
    # 2) Interface displacement ξ(y) and derivative
    #    ξ(y) = Δξ tanh(y/L)
    # ---------------------------------------------------------
    ξ     = Δξ .* tanh.(y ./ L)
    sech2 = 1 ./ cosh.(y ./ L).^2
    dξdy  = Δξ .* (sech2 ./ L)
    η     = -gp/g .* ξ
    dηdy  =  -gp/g .* dξdy

    # Change due to interface (pure internal thickness mode)
    @. h1_1d += (η - ξ)
    @. h2_1d += ξ

    # ---------------------------------------------------------
    # 3) Coriolis field (variable f(lat))
    # ---------------------------------------------------------
    f = 2 .* Ω .* sin.(lat_h .* deg2rad)

    # ---------------------------------------------------------
    # 4) Geostrophic balance for transports (η = 0)
    #    f m1 =  h1 g  ∂ξ/∂y
    #    f m2 = -h1 g  ∂ξ/∂y − h2 g' ∂ξ/∂y
    # ---------------------------------------------------------
    # @. m1_1d =  h1_1d * g  * dξdy / f
    # @. m2_1d = (-h1_1d * g - h2_1d * gp) * dξdy / f

    @. m1_1d =  h1_1d * gp  * dξdy / f
    @. m2_1d =  FT(0.0)

    # ---------------------------------------------------------
    # 5) Expand to 2D fields (zonal invariance of base jet)
    # ---------------------------------------------------------
    h1 = Array{FT}(undef, Nx, Ny)
    h2 = Array{FT}(undef, Nx, Ny)
    m1 = Array{FT}(undef, Nx, Ny)
    m2 = Array{FT}(undef, Nx, Ny)

    @inbounds for j in 1:Ny
        h1[:, j] .= h1_1d[j]
        h2[:, j] .= h2_1d[j]
        m1[:, j] .= m1_1d[j]
        m2[:, j] .= m2_1d[j]
    end

    # ---------------------------------------------------------
    # 6) 2D Ld-scale harmonic "noise" (baroclinic), PERIODIC IN x
    #    - deformation radius uses f0 at lat0
    #    - choose integer n_mode_x so kx Lx = 2π n (periodic)
    # ---------------------------------------------------------
    if noise_amp != 0
        # Internal wave speed (2-layer baroclinic)
        c_int = sqrt(gp * (H1 * H2) / (H1 + H2))

        # f0 at center latitude (for deformation radius only)
        f0 = 2 * Ω * sin(lat0 * deg2rad)
        Ld = abs(c_int / f0)                 # first baroclinic deformation radius

        # Target baroclinic wavenumber magnitude
        k_mag_target = 1 / (2*Ld) #2 * FT(π) / (2*Ld)    # wavelength ≈ 5·Ld
        ky           = k_mag_target / sqrt(FT(2))  # keep original choice in y

        # 2D coordinates (x,y) in meters
        lon_h = FT(p.lon1) .+ FT.(0:Nx-1) .* FT(p.dlon)
        lon0  = FT(0.5 * (p.lon1 + p.lon2))

        x = Rearth * cos(lat0 * deg2rad) .* (lon_h .- lon0) .* deg2rad   # size Nx
        Lx = maximum(x) - minimum(x)

        # Choose integer mode number in x so kx Lx = 2π * n_mode_x (periodic)
        # but as close as possible to target k_mag_target / √2
        kx_target = k_mag_target / sqrt(FT(2))
        n_mode_x  = max(1, round(Int, kx_target * Lx / (2 * π)))
        kx        = (2 * FT(π) * FT(n_mode_x)) / Lx

        # For reference / debugging you can print or log:
        # @info "Periodic noise: n_mode_x = $n_mode_x, kx = $kx, Lx = $Lx"

        # Reshape for broadcasting
        x2d = reshape(x, Nx, 1)        # (Nx, 1)
        y2d = reshape(y, 1, Ny)        # (1, Ny)

        # Periodic in x, Ld-scale-ish in both x,y
        noise2d = noise_amp * Δξ .* cos.(kx .* x2d) .* cos.(ky .* y2d)
        # rand2d = FT(2.0) .* rand(FT, Nx, Ny) .- FT(1.0)
        # @. noise2d = noise2d + noise_amp*noise_amp * Δξ * rand2d

        # Baroclinic perturbation: preserve total H
        h1 .+= noise2d
        h2 .-= noise2d
        # Note: m1, m2 are NOT adjusted after adding noise, so the
        # perturbed state is slightly off exact geostrophy by design.
    end

    return h1, h2, m1, m2
end


"""
    initialize_baroclinic_tanh_jet!(state, p;
        Δξ=5.0, L=300e3, lat0=(p.lat1 + p.lat2)/2, noise_amp=0.05)

Initialize the model with a **baroclinically unstable tanh jet** produced
by a tilted internal interface ξ(y) and zero SSH (η = 0).

- Sets `h1 = H1 − ξ`, `h2 = H2 + ξ` (pure internal thickness mode, H constant).
- Sets zonal layer transports `m1`, `m2` in local geostrophic balance
  with ξ(y) using `build_baroclinic_tanh_jet`.
- Sets meridional layer transports `n1`, `n2` to zero.
- Recomputes `H`, `H_old`, and barotropic transports `M = m1 + m2`,
  `N = n1 + n2` to remain consistent with `h1`, `h2`, `m1`, `m2`.

This initializer does **not** enforce zero barotropic transport; `M` is
whatever results from the specified layer transports.
"""
function initialize_baroclinic_tanh_jet!(
    state::State,
    p::Params;
    Δξ::FT        = FT(10.0),
    L::FT         = FT(200e3),
    lat0::FT      = FT(0.5 * (p.lat1 + p.lat2)),
    noise_amp::FT = FT(0.1),
)
    h1_cpu, h2_cpu, m1_cpu, m2_cpu =
        build_baroclinic_tanh_jet(
            p; Δξ=Δξ, L=L, lat0=lat0, noise_amp=noise_amp
        )

    # Layer thicknesses
    copyto!(state.prog1.h, h1_cpu)
    copyto!(state.prog2.h, h2_cpu)

    # Zonal layer transports
    copyto!(state.prog1.m, m1_cpu)
    copyto!(state.prog2.m, m2_cpu)

    # Meridional transports = 0
    fill!(state.prog1.n, FT(0))
    fill!(state.prog2.n, FT(0))

    return nothing
end
