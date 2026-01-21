# run.jl — stand-alone driver script for the TwoLayerSW model.
#
# This script:
# 1. Adds the `src/` directory to `LOAD_PATH` so that Julia can find the
#    `TwoLayerSW` module.
# 2. Imports `TwoLayerSW` and, when run directly (e.g. `julia run.jl`),
#    launches a two-layer rotating shallow-water simulation.
# 3. The simulation parameters include:
#      - Grid resolution (Nx, Ny)
#      - Longitude/latitude bounds
#      - Time step (dt) and barotropic subcycles (M)
#      - Layer depths and densities (H₁, H₂, ρ₁, ρ₂)
#      - Wind parameters (u10, v10, ρ_air)
#      - Drag and viscosity coefficients (Cd₁, Cd₂, ν)
#      - Shapiro filter strength, output interval, and NetCDF filename
# 4. Output fields are written to a NetCDF file if configured.
#
# Usage:
#     julia run.jl
#
# This file is not meant to be imported; the simulation block executes only
# when `run.jl` is invoked as a script.

using Pkg

# Ensure module can be found under src/
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using TwoLayerSW

# ----------------------------------------------------------------------
# Run simulation when file is executed directly
# ----------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__

    TwoLayerSW.run_twoLayer_SW(
        # ------------------------
        # Grid & domain
        # ------------------------
        Nx   = 128,
        Ny   = 64,
        lon1 = -220.0,
        lat1 = -20.0,
        lon2 = -140.0,
        lat2 = 20.0,

        # ------------------------
        # Time stepping
        # ------------------------
        dt       = 60.0,          # baroclinic time step (s)
        M        = 120,           # barotropic subcycles
        end_time = 12 * 24 * 3600.0,   # 12 days

        # ------------------------
        # Layer properties
        # ------------------------
        H1   = 200.0,
        H2   = 3800.0,
        rho1 = 1025.0,
        rho2 = 1027.0,

        # ------------------------
        # Forcing parameters
        # ------------------------
        rho_air = 1.2,
        u10     = 10.0,
        v10     = 0.0,
        Cd1     = 0.0,
        Cd2     = 0.0,

        # ------------------------
        # Dissipation / smoothing
        # ------------------------
        nu            = 10,
        smoothing_eps = 0.05,

        # ------------------------
        # Output
        # ------------------------
        save_interval = 3*3600, # 1 hours
        out_netcdf    = "two_layer_SW.nc",
    )

end
