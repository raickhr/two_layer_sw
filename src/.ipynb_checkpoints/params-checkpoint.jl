# ============================================================
# params.jl
#
# Parameter container and constructor for the 2-layer rotating
# shallow-water model.
#
# Assumes:
#   - `FT` is the floating-point type (Float32 or Float64)
#     defined at the module level.
# ============================================================

"""
    Params

Immutable container holding all physical & numerical parameters
for the two-layer rotating shallow-water model.

Each field includes its description, units, and default value
(from the original WHOI-style driver).
"""
struct Params
    Nx::Int            # number of grid points in x-direction               [–]   default: 256
    Ny::Int            # number of grid points in y-direction               [–]   default: 128

    dlon::FT           # longitudinal grid spacing                          [deg] derived
    dlat::FT           # latitudinal grid spacing                           [deg] derived

    lon1::FT           # western boundary longitude                         [deg] default: 0.0
    lat1::FT           # southern boundary latitude                         [deg] default: -20.0
    lon2::FT           # eastern boundary longitude                         [deg] default: 80.0
    lat2::FT           # northern boundary latitude                         [deg] default: 20.0

    earthRadius::FT    # Earth radius                                       [m]   default: 6371e3
    Ω::FT              # planetary rotation rate                            [rad s⁻¹] default: 2π/(24×3600)

    dt::FT             # baroclinic time step                               [s]   default: 10.0
    dtBT::FT           # barotropic (external) subcycle timestep            [s]   derived = dt/M
    M::Int             # number of barotropic subcycles per baroclinic step [–]   default: 180

    g::FT              # gravitational acceleration                         [m s⁻²] default: 9.81
    gp::FT             # reduced gravity g(ρ₂−ρ₁)/ρ₂                        [m s⁻²] derived

    H1::FT             # resting upper-layer thickness                       [m]   default: 200.0
    H2::FT             # resting lower-layer thickness                       [m]   default: 3800.0

    rho1::FT           # density of upper layer                              [kg m⁻³] default: 1025.0
    rho2::FT           # density of lower layer                              [kg m⁻³] default: 1027.0
    rho_air::FT        # air density                                         [kg m⁻³] default: 1.2

    hmin::FT           # minimum allowed thickness                           [m]   default: 1e-3

    Cd1::FT            # surface wind-stress drag coefficient                [–]   default: 0.05
    Cd2::FT            # bottom drag coefficient                             [–]   default: 0.05

    nu::FT             # viscosity coefficient (≈ √A₄ for biharmonic)        [m² s⁻¹] default: 1e3

    u10::FT            # prescribed zonal 10-m wind speed                    [m s⁻¹] default: 10.0
    v10::FT            # prescribed meridional 10-m wind speed               [m s⁻¹] default: 0.0

    smoothing_eps::FT  # Shapiro filter strength                             [–]   default: 0.02
end


# ============================================================
# Constructor: make_params
# ============================================================

"""
    make_params(; kwargs...) -> Params

Construct a [`Params`](@ref) object using keyword arguments.

Any keyword may override the default values from the original
two-layer model driver.

Defaults:
---------
- Nx = 256, Ny = 128
- lon1 = 0.0, lat1 = -20.0
- lon2 = 80.0, lat2 = 20.0
- earthRadius = 6371e3 m
- Ω = 2π/(24×3600)
- dt = 180 s
- M = 120
- g = 9.81 m/s²
- H1 = 200 m, H2 = 3800 m
- rho1 = 1025 kg/m³, rho2 = 1027 kg/m³
- rho_air = 1.2 kg/m³
- hmin = 1e–3 m
- Cd1 = Cd2 = 0.05
- nu = 1e3 m²/s
- u10 = 10 m/s, v10 = 0 m/s
- smoothing_eps = 0.02

Derived internally:
-------------------
- dlon = (lon2 − lon1) / (Nx − 1)
- dlat = (lat2 − lat1) / (Ny − 1)
- gp   = g(ρ₂ − ρ₁)/ρ₂
- dtBT = dt/M
"""
function make_params(; kwargs...)
    # ------------------------------------------------------------
    # Default values (single source of truth)
    # ------------------------------------------------------------
    defaults = (
        Nx = 256,
        Ny = 128,

        lon1 = 0.0,
        lat1 = -20.0,
        lon2 = 80.0,
        lat2 = 20.0,

        earthRadius = 6371e3,
        Ω = 2π / (24 * 3600),

        dt = 10.0,
        M  = 180,

        g  = 9.81,
        H1 = 200.0,
        H2 = 3800.0,

        rho1 = 1025.0,
        rho2 = 1027.0,
        rho_air = 1.2,

        hmin = 1e-3,

        Cd1 = 0.05,
        Cd2 = 0.05,

        nu  = 1,

        u10 = 10.0,
        v10 = 0.0,

        smoothing_eps = 0.02,
    )

    # ------------------------------------------------------------
    # Merge overrides (kwargs) with defaults
    # ------------------------------------------------------------
    # Using (; defaults..., kwargs...) ensures kwargs overwrite defaults
    kw = (; defaults..., kwargs...)

    # ------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------
    dlon = (kw.lon2 - kw.lon1) / (kw.Nx - 1)
    dlat = (kw.lat2 - kw.lat1) / (kw.Ny - 1)
    gp   = kw.g * (kw.rho2 - kw.rho1) / kw.rho2
    dtBT = kw.dt / kw.M

    # ------------------------------------------------------------
    # Construct Params
    # ------------------------------------------------------------
    return Params(
        kw.Nx, kw.Ny,
        FT(dlon), FT(dlat),
        FT(kw.lon1), FT(kw.lat1), FT(kw.lon2), FT(kw.lat2),
        FT(kw.earthRadius),
        FT(kw.Ω),
        FT(kw.dt),
        FT(dtBT),
        kw.M,
        FT(kw.g),
        FT(gp),
        FT(kw.H1),
        FT(kw.H2),
        FT(kw.rho1),
        FT(kw.rho2),
        FT(kw.rho_air),
        FT(kw.hmin),
        FT(kw.Cd1),
        FT(kw.Cd2),
        FT(kw.nu),
        FT(kw.u10),
        FT(kw.v10),
        FT(kw.smoothing_eps),
    )
end
