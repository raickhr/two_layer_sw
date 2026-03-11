############################################################
# advection_weno.jl
#
# True WENO-Z 5th-order flux-form advection on a 2D C-grid.
#
# This file provides:
#
#   - wenoZ5_left      : left-biased 5th-order reconstruction (Jiang & Shu 1996)
#   - wenoZ5_right     : right-biased reconstruction (via stencil reversal)
#   - wenoZ5_upwind    : velocity-based upwind selection of left/right states
#   - wenoZ_flux1d     : 1D WENO-Z face flux for a single interface
#   - k_calc_WENOZ_flux2d! :
#         GPU kernel computing -∇·(u φ) on a 2D staggered C-grid
#
# Assumptions
# -----------
# - `FT` is the floating-point type (e.g., Float32 or Float64) defined
#   at module scope.
#
# - Grid labels are Int16 constants defined elsewhere in the module:
#       const HGRID::Int16 = ...
#       const UGRID::Int16 = ...
#       const VGRID::Int16 = ...
#       const ZGRID::Int16 = ...
#
# - A device-safe `stencil` helper is available with signature
#
#       stencil(field,
#               i::Int, j::Int,
#               i_increment::Int, j_increment::Int,
#               Nx::Int, Ny::Int,
#               grid::Int16)
#
#   which:
#       * interprets (i, j) as the base discrete index,
#       * applies the (i_increment, j_increment) offset,
#       * wraps/clamps in x,y according to `grid` (HGRID/UGRID/VGRID/ZGRID),
#       * and returns `field[i_index, j_index]`.
#
# - CUDA is available (`using CUDA`) and kernels are launched via `@cuda`.
#
# Sign convention
# ---------------
# The kernel `k_calc_WENOZ_flux2d!` computes the conservative tendency
#
#       dφ[i,j] = ( -∇·(u φ) )_ij
#
# so that the caller typically updates with
#
#       φ_new .= φ_old .+ dt .* dφ
#
# to advance the field in time.
############################################################


# ============================================================
# Grid helper: where φ, u_face, v_face live
# ============================================================

"""
    get_grid_centerType(gridCenter::Int16) -> (grid_φ, grid_u, grid_v)

Return the grid-character tuple used by `stencil` for:

  - `grid_φ` : locations of the advected quantity φ,
  - `grid_u` : locations of the zonal / x-face quantity (u_face, dy_face),
  - `grid_v` : locations of the meridional / y-face quantity (v_face, dx_face).

Conventions
-----------
- `HGRID` : cell centers / mass points
- `UGRID` : u-points (east–west cell faces)
- `VGRID` : v-points (north–south cell faces)
- `ZGRID` : corner / ζ-points

Typical choices:
- `gridCenter = HGRID` (default use in most SW models):
    φ on h, u_face & dy_face on u, v_face & dx_face on v
- `gridCenter = UGRID`:
    φ on u, u_face & dy_face on h, v_face & dx_face on z
- `gridCenter = VGRID`:
    φ on v, u_face & dy_face on z, v_face & dx_face on h
"""
@inline function get_grid_centerType(gridCenter::Int16)::NTuple{3,Int16}
    if gridCenter == HGRID
        # φ on h, u_face/dy_face on u, v_face/dx_face on v
        return (HGRID, UGRID, VGRID)
    elseif gridCenter == UGRID
        # φ on u, u_face/dy_face on h, v_face/dx_face on z
        return (UGRID, HGRID, ZGRID)
    elseif gridCenter == VGRID
        # φ on v, u_face/dy_face on z, v_face/dx_face on h
        return (VGRID, ZGRID, HGRID)
    else
        # Fallback: treat unknown types like HGRID
        return (HGRID, UGRID, VGRID)
    end
end


# ============================================================
# WENO-Z 5th order core
# ============================================================

# Global smoothness epsilon (Borges et al. WENO-Z)
const WENO_EPS = FT(1e-6)

"""
    wenoZ5_left(φm2, φm1, φ0, φp1, φp2) -> FT

5th-order **left-biased** WENO-Z reconstruction at interface `i+1/2`
for flow to the right.

The arguments correspond to cell-centered stencil values:

- `φm2 = φ_{i-2}`
- `φm1 = φ_{i-1}`
- `φ0  = φ_i`
- `φp1 = φ_{i+1}`
- `φp2 = φ_{i+2}`
"""
@inline function wenoZ5_left(
    φm2::FT, φm1::FT, φ0::FT, φp1::FT, φp2::FT
)::FT
    # Linear (optimal) weights
    d0 = FT(0.1)
    d1 = FT(0.6)
    d2 = FT(0.3)

    # Candidate polynomials (Jiang & Shu 1996)
    q0 = ( FT(2)*φm2 - FT(7)*φm1 + FT(11)*φ0 ) / FT(6)  # {i-2, i-1, i}
    q1 = ( -φm1 + FT(5)*φ0 + FT(2)*φp1 ) / FT(6)       # {i-1, i, i+1}
    q2 = ( FT(2)*φ0 + FT(5)*φp1 - φp2 ) / FT(6)        # {i, i+1, i+2}

    # Smoothness indicators β_k
    β0 = ( FT(13)/FT(12) * (φm2 - FT(2)*φm1 + φ0)^2 +
           FT(1)/FT(4)  * (φm2 - FT(4)*φm1 + FT(3)*φ0)^2 )

    β1 = ( FT(13)/FT(12) * (φm1 - FT(2)*φ0 + φp1)^2 +
           FT(1)/FT(4)  * (φm1 - φp1)^2 )

    β2 = ( FT(13)/FT(12) * (φ0 - FT(2)*φp1 + φp2)^2 +
           FT(1)/FT(4)  * (FT(3)*φ0 - FT(4)*φp1 + φp2)^2 )

    # Global smoothness indicator τ_5 (Borges et al. 2008, WENO-Z)
    τ5 = abs(β0 - β2)

    # Nonlinear weights α_k, p = 2 is standard
    pz = FT(2)
    α0 = d0 * (FT(1) + (τ5 / (β0 + WENO_EPS))^pz)
    α1 = d1 * (FT(1) + (τ5 / (β1 + WENO_EPS))^pz)
    α2 = d2 * (FT(1) + (τ5 / (β2 + WENO_EPS))^pz)

    αsum = α0 + α1 + α2
    ω0 = α0 / αsum
    ω1 = α1 / αsum
    ω2 = α2 / αsum

    return ω0*q0 + ω1*q1 + ω2*q2
end

"""
    wenoZ5_right(φm2, φm1, φ0, φp1, φp2) -> FT

5th-order **right-biased** WENO-Z reconstruction at interface `i+1/2`
for flow to the left.

Implemented by calling `wenoZ5_left` on the reversed stencil
`{φp2, φp1, φ0, φm1, φm2}`.
"""
@inline function wenoZ5_right(
    φm2::FT, φm1::FT, φ0::FT, φp1::FT, φp2::FT
)::FT
    # Right state at i+1/2 for left-going waves is equivalent to
    # left state at i+1/2 for the reversed stencil.
    return wenoZ5_left(φp2, φp1, φ0, φm1, φm2)
end

"""
    wenoZ5_upwind(φm2, φm1, φ0, φp1, φp2, vel) -> FT

Velocity-based upwind selection:

- If `vel ≥ 0`, returns `wenoZ5_left(...)`.
- If `vel < 0`, returns `wenoZ5_right(...)`.
"""
@inline function wenoZ5_upwind(
    φm2::FT, φm1::FT, φ0::FT, φp1::FT, φp2::FT,
    vel::FT,
)::FT
    if vel ≥ FT(0)
        return wenoZ5_left(φm2, φm1, φ0, φp1, φp2)   # flow to the right
    else
        return wenoZ5_right(φm2, φm1, φ0, φp1, φp2)  # flow to the left
    end
end

"""
    wenoZ_flux1d(φm2, φm1, φ0, φp1, φp2,
                 normal_vel, L_face) -> FT

Compute the **WENO-Z flux at a single face** (true 5th order).

Given a 5-point stencil `{φm2, φm1, φ0, φp1, φp2}` around the interface,
and face-normal velocity `normal_vel` and face length `L_face`, this function:

1. Reconstructs the interface state `φ_face` using 5th-order WENO-Z
   with upwind direction determined by `normal_vel`.
2. Returns the physical flux

       F_face = normal_vel * φ_face * L_face
"""
@inline function wenoZ_flux1d(
    φm2::FT, φm1::FT, φ0::FT, φp1::FT, φp2::FT,
    normal_vel::FT, L_face::FT,
)::FT
    φ_face = wenoZ5_upwind(φm2, φm1, φ0, φp1, φp2, normal_vel)
    return normal_vel * φ_face * L_face
end


# ============================================================
# 2D WENO-Z flux-form advection kernel
# ============================================================

"""
    k_calc_WENOZ_flux2d!(dφ, φ, u_face, v_face,
                         dx_face, dy_face, dArea,
                         Nx, Ny, gridCenter)

GPU kernel: 2D WENO-Z advection in conservative flux form.

Computes, for each cell (i,j),

    dφ[i,j] = ( -∇·(u φ) )_ij

using **true 5th-order WENO-Z** at each face, with separate
5-point stencils centered on each interface.

Inputs
------
- `dφ      :: CuArray{FT,2}` :
      output tendency, same grid as `φ`.

- `φ       :: CuArray{FT,2}` :
      scalar to be advected, located on the grid indicated by `gridCenter`
      (one of HGRID, UGRID, VGRID).

- `u_face  :: CuArray{FT,2}` :
      face-normal velocity on x-faces (east/west of the φ-cells), residing
      on the grid returned as `grid_u` by `get_grid_centerType(gridCenter)`.

- `v_face  :: CuArray{FT,2}` :
      face-normal velocity on y-faces (north/south of the φ-cells), residing
      on the grid returned as `grid_v` by `get_grid_centerType(gridCenter)`.

- `dx_face :: CuArray{FT,2}` :
      face length in x-direction (used with v_face on `grid_v`).

- `dy_face :: CuArray{FT,2}` :
      face length in y-direction (used with u_face on `grid_u`).

- `dArea   :: CuArray{FT,2}` :
      control-volume area at φ-locations (grid `grid_φ`).

- `Nx, Ny  :: Int` :
      horizontal grid size (number of φ-cells in x and y).

- `gridCenter :: Int16` :
      grid label where φ lives:
          HGRID → φ on h, u_face on u, v_face on v
          UGRID → φ on u, u_face on h, v_face on z
          VGRID → φ on v, u_face on z, v_face on h

Boundary handling
-----------------
All boundary and staggering details are handled via `stencil(...)`,
which typically wraps periodically in x and clamps in y according to
your grid conventions.

Usage pattern
-------------
The kernel fills `dφ` with the spatial tendency:

    dφ .= ( -∇·(u φ) )

Caller then usually does:

    φ .= φ .+ dt .* dφ
"""
function k_calc_WENOZ_flux2d!(
    dφ,                 # (Nx,Ny) output: ( -∇·(u φ) )
    φ,                  # (Nx,Ny) scalar on gridCenter
    u_face, v_face,     # (Nx,Ny) face-normal velocities (zonal, meridional)
    dx_face, dy_face,   # (Nx,Ny) face lengths
    dArea,              # (Nx,Ny) control-volume area
    Nx::Int, Ny::Int,
    gridCenter::Int16,
)
    # Decide grid chars once per thread (φ-grid, u-grid, v-grid)
    grid_φ, grid_u, grid_v = get_grid_centerType(gridCenter)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        # Local control-volume area
        A = dArea[i, j]

        # -------------------------
        # X-direction contribution
        # -------------------------
        # u at faces:
        #   east face of cell (i,j):  u_face[i,  j]
        #   west face of cell (i,j):  u_face[i-1,j]
        u_east = stencil(u_face, i, j,   0, 0, Nx, Ny, grid_u)
        u_west = stencil(u_face, i, j,  -1, 0, Nx, Ny, grid_u)

        # face lengths in y-direction (height of face)
        L_east = stencil(dy_face, i, j,   0, 0, Nx, Ny, GRID0)
        L_west = stencil(dy_face, i, j,  -1, 0, Nx, Ny, GRID0)

        # East face (i+1/2): stencil {φ_{i-2},...,φ_{i+2}}
        F_east = wenoZ_flux1d(
            stencil(φ, i, j,  -2, 0, Nx, Ny, grid_φ),  # φ_{i-2}
            stencil(φ, i, j,  -1, 0, Nx, Ny, grid_φ),  # φ_{i-1}
            stencil(φ, i, j,   0, 0, Nx, Ny, grid_φ),  # φ_i
            stencil(φ, i, j,  +1, 0, Nx, Ny, grid_φ),  # φ_{i+1}
            stencil(φ, i, j,  +2, 0, Nx, Ny, grid_φ),  # φ_{i+2}
            u_east, L_east,
        )

        # West face (i-1/2): stencil {φ_{i-3},...,φ_{i+1}}
        F_west = wenoZ_flux1d(
            stencil(φ, i, j,  -3, 0, Nx, Ny, grid_φ),  # φ_{i-3}
            stencil(φ, i, j,  -2, 0, Nx, Ny, grid_φ),  # φ_{i-2}
            stencil(φ, i, j,  -1, 0, Nx, Ny, grid_φ),  # φ_{i-1}
            stencil(φ, i, j,   0, 0, Nx, Ny, grid_φ),  # φ_i
            stencil(φ, i, j,  +1, 0, Nx, Ny, grid_φ),  # φ_{i+1}
            u_west, L_west,
        )

        Xcontrib = (F_west - F_east) / A

        # -------------------------
        # Y-direction contribution
        # -------------------------
        # v at faces:
        #   north face of cell (i,j): v_face[i,  j]
        #   south face of cell (i,j): v_face[i,  j-1]
        v_north = stencil(v_face, i, j,  0,  0, Nx, Ny, grid_v)
        v_south = stencil(v_face, i, j,  0, -1, Nx, Ny, grid_v)

        # face lengths in x-direction (width of face)
        L_north = stencil(dx_face, i, j,  0,  0, Nx, Ny, GRID0)
        L_south = stencil(dx_face, i, j,  0, -1, Nx, Ny, GRID0)

        # North face (j+1/2): stencil {φ_{j-2},...,φ_{j+2}}
        F_north = wenoZ_flux1d(
            stencil(φ, i, j,  0, -2, Nx, Ny, grid_φ),  # φ_{j-2}
            stencil(φ, i, j,  0, -1, Nx, Ny, grid_φ),  # φ_{j-1}
            stencil(φ, i, j,  0,  0, Nx, Ny, grid_φ),  # φ_j
            stencil(φ, i, j,  0, +1, Nx, Ny, grid_φ),  # φ_{j+1}
            stencil(φ, i, j,  0, +2, Nx, Ny, grid_φ),  # φ_{j+2}
            v_north, L_north,
        )

        # South face (j-1/2): stencil {φ_{j-3},...,φ_{j+1}}
        F_south = wenoZ_flux1d(
            stencil(φ, i, j,  0, -3, Nx, Ny, grid_φ),  # φ_{j-3}
            stencil(φ, i, j,  0, -2, Nx, Ny, grid_φ),  # φ_{j-2}
            stencil(φ, i, j,  0, -1, Nx, Ny, grid_φ),  # φ_{j-1}
            stencil(φ, i, j,  0,  0, Nx, Ny, grid_φ),  # φ_j
            stencil(φ, i, j,  0, +1, Nx, Ny, grid_φ),  # φ_{j+1}
            v_south, L_south,
        )

        Ycontrib = (F_south - F_north) / A

        # Total contribution: ( -∇·(u φ) )
        dφ[i, j] = Xcontrib + Ycontrib
    end

    return
end
