module TwoLayerSW

# ============================================================
# TwoLayerSW.jl
#
# Main module for the GPU-accelerated 2-layer rotating
# shallow-water model on a C-grid.
#
# Directory layout (for reference):
#
#   src/
#     TwoLayerSW.jl               # this file (main module)
#     params.jl                   # Params and basic constants
#     grid.jl                     # grid construction
#     stateVars.jl                # State, Prognostic, Forcing, Temporary
#     reconstruct.jl              # C-grid reconstructions & face velocities
#     utils.jl                    # helper utilities
#     initial_condition.jl        # initial-condition builders
#     advection_weno.jl           # WENO-Z machinery + flux kernels
#     forcing.jl                  # wind stress & bottom drag
#     coriolis_curvature.jl       # Coriolis & metric terms
#     filters.jl                  # Shapiro and other filters
#     boundaries.jl               # wall / periodic boundary conditions
#     barotropic.jl               # external-mode time stepping
#     baroclinic.jl               # internal-mode time stepping (RK4 etc.)
#     io_netcdf.jl                # NetCDF init/append/output helpers
#     driver.jl                   # high-level `run_twoLayer_SW` driver
#
# Top-level API:
#   - `Params`           : immutable parameter container
#   - `run_twoLayer_SW`  : main driver to configure and run a simulation
# ============================================================

using CUDA
using NCDatasets
using Statistics
using Random
using ProgressMeter

CUDA.allowscalar(false)

const FT = Float32

# Core configuration & state
include("params.jl")
include("grid.jl")
include("stateVars.jl")

# Numerics: reconstructions, utilities, ICs, advection
include("reconstruct.jl")
include("utils.jl")
include("initial_condition.jl")
include("advection_weno.jl")

# Forcing & physics
include("forcing.jl")
include("coriolis_curvature.jl")
include("filters.jl")
include("boundaries.jl")
include("barotropic.jl")
include("baroclinic.jl")

# I/O and driver
include("io_netcdf.jl")
include("driver.jl")

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
export Params, run_twoLayer_SW

end # module TwoLayerSW
