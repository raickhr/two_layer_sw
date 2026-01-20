"""
advection_weno.jl

WENO-Z 5th order flux-form advection on a 2D C-grid.

This file provides:

- `wenoZ5_left`   : left-biased 5th-order reconstruction (Jiang & Shu 1996)
- `wenoZ5_right`  : right-biased reconstruction (using stencil reversal)
- `wenoZ5_upwind` : velocity-based upwind selection of left/right states
- `wenoZ_flux1d`  : 1D flux-form WENO-Z contribution for a single cell
- `k_calc_WENOZ_flux2d!` : GPU kernel computing   -∇·(u φ)  on a 2D grid

Assumptions
-----------
- `FT` is the floating-point type (e.g. Float32/Float64) defined at module scope.
- `iper(i, Nx)` implements **periodic** wrap in x.
- `clamp1(j, Ny)` implements clamping to `[1, Ny]` in y (solid walls).
- `CUDA` is available (`using CUDA`) and kernels are launched via `@cuda`.

Sign convention
---------------
`k_calc_WENOZ_flux2d!` returns

    dφ[i,j] = ( -∇·(u φ) )_ij

so that, in the caller, you typically do

    φ_new = φ_old + dt * dφ

for an advective update in conservative flux form.
"""

# ===========================
# WENO-Z 5th order core
# ===========================

const WENO_EPS = FT(1e-6)

"""
    wenoZ5_left(φm2, φm1, φ0, φp1, φp2) -> FT

5th-order **left-biased** WENO-Z reconstruction at interface `i+1/2`
for flow to the right.

Arguments correspond to cell-centered stencil values:
- `φm2 = φ_{i-2}`
- `φm1 = φ_{i-1}`
- `φ0  = φ_i`
- `φp1 = φ_{i+1}`
- `φp2 = φ_{i+2}`
"""
@inline function wenoZ5_left(φm2::FT, φm1::FT, φ0::FT, φp1::FT, φp2::FT)::FT
    # Linear weights (optimal)
    d0 = FT(0.1)
    d1 = FT(0.6)
    d2 = FT(0.3)

    # Candidate reconstructions (Jiang & Shu 1996)
    q0 = ( FT(2)*φm2 - FT(7)*φm1 + FT(11)*φ0 ) / FT(6)   # stencil {i-2,i-1,i}
    q1 = ( -φm1 + FT(5)*φ0 + FT(2)*φp1 ) / FT(6)        # stencil {i-1,i,i+1}
    q2 = ( FT(2)*φ0 + FT(5)*φp1 - φp2 ) / FT(6)         # stencil {i,i+1,i+2}

    # Smoothness indicators β_k
    β0 = ( FT(13)/FT(12) * (φm2 - FT(2)*φm1 + φ0)^2 +
           FT(1)/FT(4)  * (φm2 - FT(4)*φm1 + FT(3)*φ0)^2 )

    β1 = ( FT(13)/FT(12) * (φm1 - FT(2)*φ0 + φp1)^2 +
           FT(1)/FT(4)  * (φm1 - φp1)^2 )

    β2 = ( FT(13)/FT(12) * (φ0 - FT(2)*φp1 + φp2)^2 +
           FT(1)/FT(4)  * (FT(3)*φ0 - FT(4)*φp1 + φp2)^2 )

    # Borges et al. (WENO-Z) global smoothness indicator
    τ5 = abs(β0 - β2)

    # Nonlinear weights α_k (p = 2 is standard)
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
@inline function wenoZ5_right(φm2::FT, φm1::FT, φ0::FT, φp1::FT, φp2::FT)::FT
    # Right state at i+1/2 for left-going waves is equivalent to
    # left state at i+1/2 for the reversed stencil.
    return wenoZ5_left(φp2, φp1, φ0, φm1, φm2)
end

"""
    wenoZ5_upwind(φm2, φm1, φ0, φp1, φp2, vel) -> FT

Velocity-based upwind selection:

- If `vel ≥ 0`, returns `wenoZ5_left(...)`
- Otherwise, returns `wenoZ5_right(...)`
"""
@inline function wenoZ5_upwind(φm2::FT, φm1::FT, φ0::FT, φp1::FT, φp2::FT,
                               vel::FT)::FT
    if vel ≥ FT(0)
        return wenoZ5_left(φm2, φm1, φ0, φp1, φp2)   # flow to the right
    else
        return wenoZ5_right(φm2, φm1, φ0, φp1, φp2)  # flow to the left
    end
end

"""
    wenoZ_flux1d(φm2, φm1, φ0, φp1, φp2,
                 u_west, u_east,
                 L_west, L_east,
                 A) -> FT

Compute **1D flux-form WENO-Z contribution** for a single cell:

Returns
-------
`dφ_i = (F_{i-1/2} - F_{i+1/2}) / A`

where
- Interface states are reconstructed with WENO-Z (upwinded)
- Fluxes are `F = u * φ * L`, with:
    * `u_west`, `u_east` : velocities at west/east faces
    * `L_west`, `L_east` : face lengths (dy or dx)
- `A` is the cell area
"""
@inline function wenoZ_flux1d(
    φm2::FT, φm1::FT, φ0::FT, φp1::FT, φp2::FT,
    u_west::FT, u_east::FT,
    L_west::FT, L_east::FT,
    A::FT)::FT

    # Interface states
    φ_west  = wenoZ5_upwind(φm2, φm1, φ0, φp1, φp2, u_west)
    φ_east  = wenoZ5_upwind(φm2, φm1, φ0, φp1, φp2, u_east)

    # Physical fluxes F = u * φ * L (scalar advection)
    F_west  = u_west * φ_west  * L_west
    F_east  = u_east * φ_east  * L_east

    # (F_{i-1/2} - F_{i+1/2}) / A
    return (F_west - F_east) / A
end


"""
    k_calc_WENOZ_flux2d!(dφ, φ, u_face, v_face,
                         dx_face, dy_face, dArea,
                         Nx, Ny)

GPU kernel: 2D WENO-Z 5th-order advection in conservative flux form.

Computes, for each cell (i,j),
    dφ[i,j] = ( -∇·(u φ) )_ij

Inputs
------
- `φ      :: CuArray{FT,2}` : cell-centered scalar on h-cells
- `u_face :: CuArray{FT,2}` : face-normal velocity on EAST faces of h-cells
- `v_face :: CuArray{FT,2}` : face-normal velocity on NORTH faces of h-cells
- `dx_face, dy_face`       : face lengths (same staggering as `u_face`, `v_face`)
- `dArea`                  : cell area at h-points
- `Nx, Ny`                 : horizontal grid size

Boundary handling
-----------------
- Periodic in x via `iper`
- Clamped in y via `clamp1`, consistent with solid-wall north/south boundaries

Usage pattern
-------------
The kernel fills `dφ` with the **spatial** tendency:

    dφ = ( -∇·(u φ) )

Caller should update:

    φ .= φ .+ dt .* dφ
"""
function k_calc_WENOZ_flux2d!(
    dφ,                 # (Nx,Ny) output: ( -∇·(u φ) )
    φ,                  # (Nx,Ny) cell-centered scalar
    u_face, v_face,     # (Nx,Ny) face-normal velocities:
                        #   u_face: on east/west faces of h-cell
                        #   v_face: on north/south faces of h-cell
    dx_face, dy_face,   # (Nx,Ny) face lengths
    dArea,              # (Nx,Ny) cell area
    Nx::Int, Ny::Int)
    """
    WENO-Z 5th order 2D advection kernel in flux-form.

    Assumes:
    - Periodic in x (using iper) and clamped in y (using clamp1)
    - u_face[i,j]: velocity on EAST face of cell (i-1,j) / WEST of cell (i,j)
    - v_face[i,j]: velocity on NORTH face of cell (i,j-1) / SOUTH of cell (i,j)
    """

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        # indices for stencil
        im2 = iper(i-2, Nx)
        im1 = iper(i-1, Nx)
        ip1 = iper(i+1, Nx)
        ip2 = iper(i+2, Nx)

        jm2 = clamp1(j-2, Ny)
        jm1 = clamp1(j-1, Ny)
        jp1 = clamp1(j+1, Ny)
        jp2 = clamp1(j+2, Ny)

        A = dArea[i,j]

        # -------------------------
        # X-direction contribution
        # -------------------------
        # u at faces:
        #   east face of cell i:  u_face[i,  j]
        #   west face of cell i:  u_face[im1,j]
        u_east = u_face[i,  j]
        u_west = u_face[im1,j]

        # face lengths in y-direction (height of face)
        L_east = dy_face[i,  j]
        L_west = dy_face[im1,j]

        Xcontrib = wenoZ_flux1d(
            φ[im2, j], φ[im1, j], φ[i, j], φ[ip1, j], φ[ip2, j],
            u_west, u_east,
            L_west, L_east,
            A
        )

        # -------------------------
        # Y-direction contribution
        # -------------------------
        # v at faces:
        #   north face of cell i: v_face[i,  j]
        #   south face of cell i: v_face[i,  jm1]
        v_north = v_face[i,  j]
        v_south = v_face[i,  jm1]

        # face lengths in x-direction (width of face)
        L_north = dx_face[i,  j]
        L_south = dx_face[i,  jm1]

        Ycontrib = wenoZ_flux1d(
            φ[i, jm2], φ[i, jm1], φ[i, j], φ[i, jp1], φ[i, jp2],
            v_south, v_north,
            L_south, L_north,
            A
        )

        # Total contribution: ( -∇·(u φ) )
        dφ[i,j] = Xcontrib + Ycontrib
    end
    return
end
