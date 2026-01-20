"""
Utility functions and GPU kernels for the rotating shallow-water C-grid model.

Provides:
- `coriolis_at_lat` : Coriolis parameter f = 2Ω sin(lat)
- `calcCourant`     : Courant number using FV geometry
- `k_calc_gradient!` :
      Face-centered gradients ∂φ/∂x (east face) and ∂φ/∂y (north face)
- `k_calc_laplacian!` :
      Finite-volume ∇·(ν∇φ) diffusion operator
- `k_calc_laplacian_transport!` :
      Same as above but returns dt ⋅ ∇·(ν∇φ)

Assumptions:
- Scalars at cell centers (h-points); gradients on east/north faces
- `iper` gives periodic wrap in x; `clamp1` enforces solid walls in y
- All functions are CUDA-GPU safe (no allocations, simple arithmetic)

All Laplacian and gradient operators follow standard flux-form FV conventions.
"""


@inline coriolis_at_lat(Ω::FT, latInDeg::FT) =
    FT(2) * Ω * sin(latInDeg * (FT(pi) / FT(180)))

@inline calcCourant(vel::FT, dy::FT, dArea::FT, dt::FT) =
    vel * dy / dArea * dt

"""
Compute gradients from cell center to east face and north face.

gradx_face[i,j] ≈ (phi[i+1,j] - phi[i,j]) / dx_n2n[i,j]
grady_face[i,j] ≈ (phi[i,j+1] - phi[i,j]) / dy_n2n[i,j]

Periodic in x via `iper`, clamped in y via `clamp1`.
"""
function k_calc_gradient!(gradx_face, grady_face, phi, dx_n2n, dy_n2n, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        ip1 = iper(i + 1, Nx)
        jp1 = clamp1(j + 1, Ny)

        gradx_face[i, j] = (phi[ip1, j] - phi[i, j]) / dx_n2n[i, j]
        grady_face[i, j] = (phi[i, jp1] - phi[i, j]) / dy_n2n[i, j]
    end
    return
end

"""
Compute ∇⋅(ν ∇q) in FV form, where q is a transport (e.g. hu or hv).
"""
function k_calc_laplacian!(dHu, gradx_face, grady_face, dx_face, dy_face, dArea, nu::FT, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        im1 = iper(i - 1, Nx)
        jm1 = clamp1(j - 1, Ny)

        A = dArea[i, j]

        Fe = nu * gradx_face[i,   j] * dy_face[i,   j]
        Fw = nu * gradx_face[im1, j] * dy_face[im1, j]
        Fn = nu * grady_face[i,   j] * dx_face[i,   j]
        Fs = nu * grady_face[i,  jm1] * dx_face[i,  jm1]

        dHu[i, j] = ((Fe - Fw) + (Fn - Fs)) / A
    end
    return
end

"""
Compute dt * ∇⋅(ν ∇q) in FV form.
"""
function k_calc_laplacian_transport!(dHu, gradx_face, grady_face, dx_face, dy_face, dArea,
                                     nu::FT, Nx::Int, Ny::Int, dt::FT)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        im1 = iper(i - 1, Nx)
        jm1 = clamp1(j - 1, Ny)

        A = dArea[i, j]

        Fe = nu * gradx_face[i,   j] * dy_face[i,   j]
        Fw = nu * gradx_face[im1, j] * dy_face[im1, j]
        Fn = nu * grady_face[i,   j] * dx_face[i,   j]
        Fs = nu * grady_face[i,  jm1] * dx_face[i,  jm1]

        dHu[i, j] = dt * ((Fe - Fw) + (Fn - Fs)) / A
    end
    return
end

