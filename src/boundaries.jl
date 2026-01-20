# boundaries.jl
#
# Boundary-condition utilities and GPU kernels for the
# 2-layer rotating shallow-water model (C-grid).
#
# Provides
# --------
# Index helpers
#   - iper(i, N)    : periodic wrap in x
#   - clamp1(i, N)  : clamp index to y-boundary
#
# Meridional (N/S) wall conditions
#   - k_apply_walls_v!      : no-normal-flow (v = 0) at N/S walls
#   - k_freeslip_walls_u!   : ∂u/∂y = 0 at N/S walls (tangential free-slip)
#   - k_noslip_walls_u!     : u = 0 at N/S walls (tangential no-slip)
#
# Zonal (W/E) wall conditions
#   - k_apply_walls_u_x!    : no-normal-flow (u = 0) at W/E walls
#   - k_freeslip_walls_v_x! : ∂v/∂x = 0 at W/E walls (tangential free-slip)
#   - k_noslip_walls_v_x!   : v = 0 at W/E walls (tangential no-slip)
#
# Open / radiation boundaries (Orlanski-like, scalar)
#   - k_orlanski_open_west!  / k_orlanski_open_east!  : W/E open BC
#   - k_orlanski_open_south! / k_orlanski_open_north! : S/N open BC
#   - apply_open_boundary_x! : wrapper for W/E
#   - apply_open_boundary_y! : wrapper for S/N
#
# Sponge / relaxation layers
#   - k_sponge_relax!        : relax scalar toward reference with mask
#   - apply_sponge_layer!    : wrapper over k_sponge_relax!
#
# Assumptions
# -----------
# - `FT` is the floating type (Float32 or Float64) defined in the parent module.
# - CUDA is imported in the parent module (`using CUDA`).
# - All kernels are allocation-free and GPU-safe.

# ============================================================
# Indexing helpers
# ============================================================

"""
    iper(i, N)

Periodic wrap in the x-direction.
Maps any integer `i` to the range `1:N`.

Examples:
    iper(N + 1, N) == 1
    iper(0,     N) == N
"""
@inline iper(i, N) = mod(i - 1, N) + 1

"""
    clamp1(i, N)

Clamp index `i` to the closed interval `1:N`.

Useful for meridional (y) boundaries where no periodicity exists.
"""
@inline clamp1(i, N) = max(1, min(N, i))

# ============================================================
# Meridional wall BCs (v-normal, u-tangential)
# ============================================================

"""
    k_apply_walls_v!(n, Nx, Ny)

Enforce no-normal-flow at north/south walls for meridional
transport `n` (v-points):

    n[:, 1]  = 0
    n[:, Ny] = 0
"""
function k_apply_walls_v!(n, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx
        n[i, 1]  = FT(0)
        n[i, Ny] = FT(0)
    end
    return
end


"""
    k_apply_walls_h!(h, Nx, Ny)

Enforce zero gradient at north/south walls for h (h-points):

    h[:, 1]  = 0
    h[:, Ny] = 0
"""
function k_apply_walls_h!(h, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx
        h[i, 1]  = h[i, 2]
        # h[i, Ny] = h[i, Ny-1]
    end
    return
end

"""
    k_freeslip_walls_u!(m, Nx, Ny)

Free-slip condition at north/south walls for zonal transport `m`
(u-points), enforcing ∂u/∂y = 0:

    m[:, 1]  = m[:, 2]
    m[:, Ny] = m[:, Ny-1]
"""
function k_freeslip_walls_u!(m, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx && Ny >= 2
        m[i, 1]  = m[i, 2]
        m[i, Ny] = m[i, Ny - 1]
    end
    return
end

"""
    k_noslip_walls_u!(m, Nx, Ny)

No-slip condition at north/south walls for zonal transport `m`
(u-points), enforcing u = 0:

    m[:, 1]  = 0
    m[:, Ny] = 0
"""
function k_noslip_walls_u!(m, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx
        m[i, 1]  = FT(0)
        m[i, Ny] = FT(0)
    end
    return
end

# ============================================================
# Zonal wall BCs (u-normal, v-tangential)
# ============================================================

"""
    k_apply_walls_u_x!(m, Nx, Ny)

No-normal-flow at west/east walls for u-transport `m`
(u is normal to x-walls):

    m[1,  :] = 0
    m[Nx, :] = 0
"""
function k_apply_walls_u_x!(m, Nx::Int, Ny::Int)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= Ny
        m[1,  j] = FT(0)
        m[Nx, j] = FT(0)
    end
    return
end

"""
    k_freeslip_walls_v_x!(n, Nx, Ny)

Free-slip tangential BC for meridional transport `n` at west/east walls,
enforcing ∂v/∂x = 0:

    n[1,  :] = n[2,    :]
    n[Nx, :] = n[Nx-1, :]
"""
function k_freeslip_walls_v_x!(n, Nx::Int, Ny::Int)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= Ny && Nx >= 2
        n[1,  j] = n[2,      j]
        n[Nx, j] = n[Nx - 1, j]
    end
    return
end

"""
    k_noslip_walls_v_x!(n, Nx, Ny)

No-slip tangential BC for meridional transport `n` at west/east walls,
enforcing v = 0:

    n[1,  :] = 0
    n[Nx, :] = 0
"""
function k_noslip_walls_v_x!(n, Nx::Int, Ny::Int)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= Ny
        n[1,  j] = FT(0)
        n[Nx, j] = FT(0)
    end
    return
end

# ============================================================
# Convenience wrappers for wall BCs
# ============================================================

"""
    apply_meridional_walls!(n, Nx, Ny; threads1, blocks1)

Apply no-normal-flow wall BC for meridional transport `n` (v-points)
at north/south boundaries.
"""
function apply_meridional_walls!(
    n,
    Nx::Int,
    Ny::Int;
    threads1::Int,
    blocks1::Int,
)
    @cuda threads=threads1 blocks=blocks1 k_apply_walls_v!(n, Nx, Ny)
    return nothing
end

"""
    apply_tangential_walls_u!(m, Nx, Ny; bc=:freeslip, threads1, blocks1)

Apply tangential BC at north/south walls for zonal transport `m`.

Keyword `bc` options:
    - :freeslip  → ∂u/∂y = 0
    - :noslip    → u = 0
"""
function apply_tangential_walls_u!(
    m,
    Nx::Int,
    Ny::Int;
    bc::Symbol = :freeslip,
    threads1::Int,
    blocks1::Int,
)
    if bc === :freeslip
        @cuda threads=threads1 blocks=blocks1 k_freeslip_walls_u!(m, Nx, Ny)
    elseif bc === :noslip
        @cuda threads=threads1 blocks=blocks1 k_noslip_walls_u!(m, Nx, Ny)
    else
        error("Unknown tangential BC: $bc. Use :freeslip or :noslip.")
    end
    return nothing
end

"""
    apply_zonal_walls!(m, n, Nx, Ny;
                      normal_bc=:wall,
                      tangential_bc=:freeslip,
                      threads1, blocks1)

Apply BCs at west/east boundaries:

Normal u-transport `m`:
    normal_bc = :wall → u = 0 at x = W,E

Tangential v-transport `n`:
    tangential_bc = :freeslip → ∂v/∂x = 0
    tangential_bc = :noslip   → v = 0
"""
function apply_zonal_walls!(
    m,
    n,
    Nx::Int,
    Ny::Int;
    normal_bc::Symbol = :wall,
    tangential_bc::Symbol = :freeslip,
    threads1::Int,
    blocks1::Int,
)
    if normal_bc === :wall
        @cuda threads=threads1 blocks=blocks1 k_apply_walls_u_x!(m, Nx, Ny)
    else
        error("Unknown normal BC at x-walls: $normal_bc. Only :wall supported for now.")
    end

    if tangential_bc === :freeslip
        @cuda threads=threads1 blocks=blocks1 k_freeslip_walls_v_x!(n, Nx, Ny)
    elseif tangential_bc === :noslip
        @cuda threads=threads1 blocks=blocks1 k_noslip_walls_v_x!(n, Nx, Ny)
    else
        error("Unknown tangential BC at x-walls: $tangential_bc. Use :freeslip or :noslip.")
    end

    return nothing
end

# ============================================================
# Orlanski-type open / radiation BCs in x (scalar)
# ============================================================

"""
    k_orlanski_open_west!(φ, Nx, Ny, c, dx, dt)

Orlanski-type radiation BC at west boundary (i=1) for scalar φ:

    φ₁ⁿ⁺¹ = φ₂ⁿ - c*(dt/dx)*(φ₃ⁿ - φ₂ⁿ)

Assumes `c > 0` is an outward phase speed and `Nx ≥ 3`.
"""
function k_orlanski_open_west!(
    φ,
    Nx::Int,
    Ny::Int,
    c::FT,
    dx::FT,
    dt::FT,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= Ny && Nx >= 3
        φ2      = φ[2, j]
        φ3      = φ[3, j]
        φ[1, j] = φ2 - c * (dt / dx) * (φ3 - φ2)
    end
    return
end

"""
    k_orlanski_open_east!(φ, Nx, Ny, c, dx, dt)

Orlanski-type radiation BC at east boundary (i=Nx) for scalar φ:

    φ_Nⁿ⁺¹ = φ_{N-1}ⁿ + c*(dt/dx)*(φ_{N-1}ⁿ - φ_{N-2}ⁿ)

Assumes `c > 0` is an outward phase speed and `Nx ≥ 3`.
"""
function k_orlanski_open_east!(
    φ,
    Nx::Int,
    Ny::Int,
    c::FT,
    dx::FT,
    dt::FT,
)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= Ny && Nx >= 3
        φNm1      = φ[Nx - 1, j]
        φNm2      = φ[Nx - 2, j]
        φ[Nx, j]  = φNm1 + c * (dt / dx) * (φNm1 - φNm2)
    end
    return
end

"""
    apply_open_boundary_x!(φ, Nx, Ny; side, c, dx, dt, threads1, blocks1)

Apply Orlanski-type open boundary in x for scalar φ.

Arguments:
    side    : :west or :east
    c       : outward phase speed (FT)
    dx      : grid spacing (FT, assumed constant)
    dt      : timestep (FT)
"""
function apply_open_boundary_x!(
    φ,
    Nx::Int,
    Ny::Int;
    side::Symbol,
    c::FT,
    dx::FT,
    dt::FT,
    threads1::Int,
    blocks1::Int,
)
    if side === :west
        @cuda threads=threads1 blocks=blocks1 k_orlanski_open_west!(φ, Nx, Ny, c, dx, dt)
    elseif side === :east
        @cuda threads=threads1 blocks=blocks1 k_orlanski_open_east!(φ, Nx, Ny, c, dx, dt)
    else
        error("Unknown side for open boundary (x): $side. Use :west or :east.")
    end
    return nothing
end

# ============================================================
# Orlanski-type open / radiation BCs in y (scalar)
# ============================================================

"""
    k_orlanski_open_south!(φ, Nx, Ny, c, dy, dt)

Orlanski-type radiation BC at south boundary (j=1) for scalar φ:

    φ₁ⁿ⁺¹ = φ₂ⁿ - c*(dt/dy)*(φ₃ⁿ - φ₂ⁿ)

Assumes `c > 0` and `Ny ≥ 3`.
"""
function k_orlanski_open_south!(
    φ,
    Nx::Int,
    Ny::Int,
    c::FT,
    dy::FT,
    dt::FT,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx && Ny >= 3
        φ2      = φ[i, 2]
        φ3      = φ[i, 3]
        φ[i, 1] = φ2 - c * (dt / dy) * (φ3 - φ2)
    end
    return
end

"""
    k_orlanski_open_north!(φ, Nx, Ny, c, dy, dt)

Orlanski-type radiation BC at north boundary (j=Ny) for scalar φ:

    φ_Nⁿ⁺¹ = φ_{N-1}ⁿ + c*(dt/dy)*(φ_{N-1}ⁿ - φ_{N-2}ⁿ)

Assumes `c > 0` and `Ny ≥ 3`.
"""
function k_orlanski_open_north!(
    φ,
    Nx::Int,
    Ny::Int,
    c::FT,
    dy::FT,
    dt::FT,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx && Ny >= 3
        φNm1      = φ[i, Ny - 1]
        φNm2      = φ[i, Ny - 2]
        φ[i, Ny]  = φNm1 + c * (dt / dy) * (φNm1 - φNm2)
    end
    return
end

"""
    apply_open_boundary_y!(φ, Nx, Ny; side, c, dy, dt, threads1, blocks1)

Apply Orlanski-type open boundary in y for scalar φ.

Arguments:
    side    : :south or :north
    c       : outward phase speed (FT)
    dy      : grid spacing (FT, assumed constant)
    dt      : timestep (FT)
"""
function apply_open_boundary_y!(
    φ,
    Nx::Int,
    Ny::Int;
    side::Symbol,
    c::FT,
    dy::FT,
    dt::FT,
    threads1::Int,
    blocks1::Int,
)
    if side === :south
        @cuda threads=threads1 blocks=blocks1 k_orlanski_open_south!(φ, Nx, Ny, c, dy, dt)
    elseif side === :north
        @cuda threads=threads1 blocks=blocks1 k_orlanski_open_north!(φ, Nx, Ny, c, dy, dt)
    else
        error("Unknown side for open boundary (y): $side. Use :south or :north.")
    end
    return nothing
end

# ============================================================
# Sponge / relaxation layer
# ============================================================

"""
    k_sponge_relax!(φ, φ_ref, mask, Nx, Ny, dt, τ)

Relax scalar φ toward reference φ_ref with mask strength `mask`:

    φ ← φ + (dt/τ) * mask * (φ_ref - φ)

`mask` should be in [0, 1].
`τ` is a relaxation timescale (seconds).
"""
function k_sponge_relax!(
    φ,
    φ_ref,
    mask,
    Nx::Int,
    Ny::Int,
    dt::FT,
    τ::FT,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        w = mask[i, j]
        if w != FT(0)
            φ_ij   = φ[i, j]
            φeq    = φ_ref[i, j]
            relax  = (dt / τ) * w
            φ[i, j] = φ_ij + relax * (φeq - φ_ij)
        end
    end
    return
end

"""
    apply_sponge_layer!(φ, φ_ref, mask, Nx, Ny; dt, τ, threads2, blocks2)

Apply sponge relaxation for scalar field `φ`.

Arguments:
    φ_ref    : reference field
    mask     : strength in [0, 1]
    dt       : timestep (FT)
    τ        : relaxation timescale (FT)
"""
function apply_sponge_layer!(
    φ,
    φ_ref,
    mask,
    Nx::Int,
    Ny::Int;
    dt::FT,
    τ::FT,
    threads2::NTuple{2, Int},
    blocks2::NTuple{2, Int},
)
    @cuda threads=threads2 blocks=blocks2 k_sponge_relax!(
        φ, φ_ref, mask,
        Nx, Ny,
        dt, τ
    )
    return nothing
end
