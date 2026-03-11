# boundaries.jl
#
# Boundary-condition utilities and GPU kernels for the
# 2-layer rotating shallow-water model (C-grid).
#
# Meridional staggering (y-direction)
# -----------------------------------
# All fields are stored as (Nx, Ny), but their *physical* y-locations differ:
#
#   - v(i,1)   lies on the SOUTH boundary
#   - v(i,Ny)  lies on the NORTH boundary
#
#   - h(i,1)   is a GHOST row *below* the SOUTH boundary
#              → physical h starts at j = 2, interior up to ~north side
#
#   - u(i,1)   is a GHOST row *below* the SOUTH boundary
#              → physical u starts at j = 2
#
# There is no explicit ghost row above the north wall; h and u near the
# north boundary are interior points just inside v(i,Ny).
#
# Consequences:
#   * normal component at N/S walls is v:
#       - enforce v(i,1)  = 0 (south wall)
#       - enforce v(i,Ny) = 0 (north wall)
#
#   * row j = 1 of h and u is ghost:
#       - enforce Neumann / slip conditions there by copying from j = 2
#       - derivative stencils for HGRID/UGRID must never read j = 1 as interior
#
# Provided
# --------
# Index helpers
#   - iper(i, N)    : periodic wrap in x
#   - clamp1(i, N)  : clamp to [1, N]
#   - clamp2(i, N)  : clamp to [2, N]
#
# Meridional BCs (N/S)
#   - k_apply_walls_v!      : v = 0 at j = 1 and j = Ny
#   - k_apply_walls_h!      : Neumann at south for h (h[:,1] = h[:,2])
#   - k_freeslip_walls_u!   : ∂u/∂y = 0 at south (m[:,1] = m[:,2])
#   - k_noslip_walls_u!     : u = 0 at south (m[:,1] = 0)
#
# Zonal BCs (W/E)
#   - k_apply_walls_u_x!    : u = 0 at i = 1, Nx
#   - k_freeslip_walls_v_x! : ∂v/∂x = 0 at i = 1, Nx
#   - k_noslip_walls_v_x!   : v = 0 at i = 1, Nx
#
# Open / radiation BCs (Orlanski-like, scalar)
#   - k_orlanski_open_west!  / k_orlanski_open_east!
#   - k_orlanski_open_south! / k_orlanski_open_north!
#   - apply_open_boundary_x! / apply_open_boundary_y!
#
# Sponge / relaxation layers
#   - k_sponge_relax!, apply_sponge_layer!
#
# Assumptions
# -----------
# - FT is the floating-point type (Float32 / Float64) defined in the parent module.
# - CUDA is imported in the parent module (`using CUDA`).
# - HGRID, UGRID, VGRID, ZGRID are Int-like constants tagging stagger types.
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
    clamp1toN(i, N)

Clamp index `i` to the closed interval `1:N`.

Safe generic clamping for fields that live physically at
all rows 1..N (e.g., v).
"""
@inline clamp1toN(i, N) = max(1, min(N, i))

"""
    clamp2toN(i, N)

Clamp index `i` to the closed interval `2:N`.

Useful for h/u grids where row 1 is a ghost cell and the
physical domain starts at j = 2. Assumes N ≥ 2 in normal usage.
"""
@inline clamp2toN(i, N) = max(2, min(N, i))




# ============================================================
# Meridional BCs (v-normal; h/u tangential with ghost at j = 1)
# ============================================================

"""
    k_apply_walls_v!(n, Nx, Ny)

Enforce no-normal-flow at north/south walls for meridional
transport `n` (v-points).

Given v(i,1) and v(i,Ny) lie on the SOUTH and NORTH boundaries:

    n[:, 1]  = 0
    n[:, Ny] = 0
"""
function k_apply_walls_v!(n, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx && Ny >= 8
        # south: ghosts 1,2,3 ; wall 4
        n[i,1] = FT(0)#-n[i,7]
        n[i,2] = FT(0)#-n[i,6]
        n[i,3] = FT(0)#-n[i,5]
        n[i,4] = FT(0)


        # north: wall Ny-4 ; ghosts Ny-3,Ny-2, Ny-1,Ny
        n[i,Ny-4] = FT(0)
        n[i,Ny-3] = FT(0)#-n[i,Ny-5]
        n[i,Ny-2] = FT(0)#-n[i,Ny-6]
        n[i,Ny-1] = FT(0)#-n[i,Ny-7]
        n[i,Ny]   = FT(0)#-n[i,Ny-8]
    end
    return
end

function k_apply_walls_u!(u, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx && Ny >= 7
        u[i,1] = u[i,5]#u[i,8]
        u[i,2] = u[i,5]#u[i,7]
        u[i,3] = u[i,5]#u[i,6]
        u[i,4] = u[i,5]

        u[i,Ny-3] = u[i,Ny-4]
        u[i,Ny-2] = u[i,Ny-4]#u[i,Ny-5]
        u[i,Ny-1] = u[i,Ny-4]#u[i,Ny-6]
        u[i,Ny]   = u[i,Ny-4]#u[i,Ny-7]
    end
    return
end

function k_apply_walls_h!(h, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx && Ny >= 7
        h[i,1] = h[i,5] #h[i,8]
        h[i,2] = h[i,5] #h[i,7]
        h[i,3] = h[i,5] #h[i,6]
        h[i,4] = h[i,5]

        h[i,Ny-3] = h[i,Ny-4]
        h[i,Ny-2] = h[i,Ny-4]#h[i,Ny-5]
        h[i,Ny-1] = h[i,Ny-4]#h[i,Ny-6]
        h[i,Ny]   = h[i,Ny-4]#h[i,Ny-7]
    end
    return
end


###############################
function k_apply_walls_v_half!(n, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx && Ny >= 9
        n[i,1] = FT(0.0) #-n[i,6]
        n[i,2] = FT(0.0) #-n[i,5]
        n[i,3] = FT(0.0) #-n[i,4]

        n[i,Ny-4] = FT(0.0) #-n[i,Ny-5]
        n[i,Ny-3] = FT(0.0) #-n[i,Ny-6]
        n[i,Ny-2] = FT(0.0) #-n[i,Ny-7]
        n[i,Ny-1] = FT(0.0) #-n[i,Ny-8]
        n[i,Ny]   = FT(0.0) #-n[i,Ny-9]
    end
    return
end

function k_apply_walls_u_half!(u, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx && Ny >= 8
        u[i,1] = u[i,5] #u[i,7]
        u[i,2] = u[i,5] #u[i,6]
        u[i,3] = u[i,5]
        
        u[i,Ny-3] = u[i,Ny-5]
        u[i,Ny-2] = u[i,Ny-5] #u[i,Ny-6]
        u[i,Ny-1] = u[i,Ny-5] #u[i,Ny-7]
        u[i,Ny]   = u[i,Ny-5] #u[i,Ny-8]
    end
    return
end

function k_apply_walls_h_half!(h, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx && Ny >= 8
        h[i,1] = h[i,5] #h[i,7]
        h[i,2] = h[i,5] #h[i,6]
        h[i,3] = h[i,5]
        
        h[i,Ny-3] = h[i,Ny-5]
        h[i,Ny-2] = h[i,Ny-5] #h[i,Ny-6]
        h[i,Ny-1] = h[i,Ny-5] #h[i,Ny-7]
        h[i,Ny]   = h[i,Ny-5] #h[i,Ny-8]
    end
    return
end



"""
    k_freeslip_walls_u!(m, Nx, Ny)

Free-slip condition at the SOUTH wall for zonal transport `m` (u-points),
using row j = 1 as a ghost. Free-slip (∂u/∂y = 0) implies:

    m[:, 1] = m[:, 2]

Row j = Ny is interior and is not modified.
"""
function k_freeslip_walls_u!(m, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx && Ny >= 2
        m[i, 1] = m[i, 2]
    end
    return
end

"""
    k_noslip_walls_u!(m, Nx, Ny)

No-slip condition at the SOUTH wall for zonal transport `m` (u-points),
using row j = 1 as a ghost representing u at the wall:

    m[:, 1] = 0

Row j = Ny is interior and is not modified.
"""
function k_noslip_walls_u!(m, Nx::Int, Ny::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= Nx && Ny >= 1
        m[i, 1] = FT(0)
    end
    return
end


# ============================================================
# Zonal wall BCs (u-normal, v-tangential) – symmetric in x
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
    if j <= Ny && Nx >= 1
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
    if j <= Ny && Nx >= 1
        n[1,  j] = FT(0)
        n[Nx, j] = FT(0)
    end
    return
end


# ============================================================
# Generic stencil helper (device-side)
# ============================================================

"""
    stencil(field, i, j, di, dj, Nx, Ny, grid)

Device-side helper to fetch a neighbor of `field` on a staggered C-grid.

Base index `(i,j)` is defined by the calling thread; we then access:

    ii = i + di
    jj = j + dj

x-index is always wrapped periodically:

    i_index = iper(ii, Nx)

y-index handling depends on stagger type:

  - grid == HGRID or UGRID:
        row j = 1 is a GHOST row (south),
        physical domain starts at j = 2:
        ⇒ j_index = clamp2(jj, Ny) ∈ [2, Ny]

  - grid == VGRID or ZGRID:
        v-points (and z-points) live physically at j = 1..Ny:
        ⇒ j_index = clamp1(jj, Ny) ∈ [1, Ny]

All this helper does is guarantee in-bounds access consistent with the
staggering. Physical BCs (no-normal-flow, slip, etc.) are enforced by
the explicit boundary-condition kernels above.
"""
@inline function stencil(
    field,
    i::Int, j::Int,
    di::Int,
    dj::Int,
    Nx::Int,
    Ny::Int,
    grid::Int16,
)
    ii = i + di
    jj = j + dj

    i_index = iper(ii, Nx)
    j_index = clamp1toN(jj, Ny)

    returnVal = field[i_index, j_index]

    return returnVal
end


# ============================================================
# Convenience wrappers for N/S & W/E BCs
# ============================================================

"""
    apply_meridional_walls!(n, Nx, Ny; threads1, blocks1)

Apply no-normal-flow BC at BOTH north and south boundaries
for meridional transport `n` (v-points):

    n[:,1]  = 0   (south wall)
    n[:,Ny] = 0   (north wall)
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

Apply tangential BC at the SOUTH boundary for zonal transport `m` (u-points),

    - :freeslip  → ∂u/∂y = 0 → m[:,1] = m[:,2]
    - :noslip    → u = 0     → m[:,1] = 0

Row j = Ny is interior and unaffected.
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
        error("Unknown tangential BC for u at south wall: $bc. Use :freeslip or :noslip.")
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

Orlanski-type radiation BC at west boundary (i = 1) for scalar φ:

    φ[1,j] ← φ[2,j] - c*(dt/dx)*(φ[3,j] - φ[2,j])

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

Orlanski-type radiation BC at east boundary (i = Nx) for scalar φ:

    φ[Nx,j] ← φ[Nx-1,j] + c*(dt/dx)*(φ[Nx-1,j] - φ[Nx-2,j])

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
    return
end


# ============================================================
# Orlanski-type open / radiation BCs in y (scalar)
# ============================================================

"""
    k_orlanski_open_south!(φ, Nx, Ny, c, dy, dt)

Orlanski-type radiation BC at south boundary (j = 1) for scalar φ:

    φ[i,1] ← φ[i,2] - c*(dt/dy)*(φ[i,3] - φ[i,2])

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

Orlanski-type radiation BC at north boundary (j = Ny) for scalar φ:

    φ[i,Ny] ← φ[i,Ny-1] + c*(dt/dy)*(φ[i,Ny-1] - φ[i,Ny-2])

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
    return
end


# ============================================================
# Sponge / relaxation layer
# ============================================================

"""
    k_sponge_relax!(φ, φ_ref, mask, Nx, Ny, dt, τ)

Relax scalar φ toward reference φ_ref with mask strength `mask`:

    φ ← φ + (dt/τ) * mask * (φ_ref - φ)

`mask` should be in [0, 1].
`τ` is a relaxation timescale (seconds, τ > 0).
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

    if i <= Nx && j <= Ny && τ > FT(0)
        w = mask[i, j]
        if w != FT(0)
            φ_ij  = φ[i, j]
            φeq   = φ_ref[i, j]
            relax = (dt / τ) * w
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
    τ        : relaxation timescale (FT, τ > 0)
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
        dt, τ,
    )
    return nothing
end