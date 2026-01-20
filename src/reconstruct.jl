# reconstruct.jl
#
# GPU kernels for C-grid reconstructions, face-velocity calculations,
# and barotropic–baroclinic mode correction.
#
# Provides:
#   - k_recon_h_in_u!        : h(h-point) -> h at u-points (x-average)
#   - k_recon_h_in_v!        : h(h-point) -> h at v-points (y-average)
#   - k_recon_u_in_v!        : u(u-point) -> u at v-points (4-pt average)
#   - k_recon_v_in_u!        : v(v-point) -> v at u-points (4-pt average)
#   - k_calc_faceVels_for_ucell! : face velocities for u-cells (advecting m)
#   - k_calc_faceVels_for_vcell! : face velocities for v-cells (advecting n)
#   - k_mode_correction!     : low-level GPU kernel for mode correction
#   - mode_correction!       : high-level wrapper callable from the driver
#
# Assumes:
#   - FT, State, Params, iper, clamp1 are defined in the module scope
#   - CUDA is available (e.g., `using CUDA` in the parent module)
#
# All kernels are allocation-free and GPU-safe.
#
# C-grid indexing convention (schematic for a single h-cell):
#
#       |---------------- v(i, j) --------------|
#       |                                       |
#       |                                       |
#       |                                       |
#   u(i-1, j)            h(i, j)              u(i, j)
#       |                                       |
#       |                                       |
#       |                                       |
#       |--------------- v(i, j-1) -------------|


# ============================================================
# Thickness reconstructions
# ============================================================

function k_recon_h_in_u!(
    h_in_u, h,
    Nx::Int, Ny::Int,
    hmin::FT,
)
    """
    Reconstruct layer thickness `h` at u-points by averaging in x:

        h_in_u(i,j) = 0.5 * [h(i,j) + h(i+1,j)]

    - Periodic in x via `iper`.
    - Enforces a minimum thickness `hmin` at u-points.
    """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        ip1 = iper(i + 1, Nx)
        hf  = FT(0.5) * (h[i, j] + h[ip1, j])
        h_in_u[i, j] = max(hf, hmin)
    end
    return
end

function k_recon_h_in_v!(
    h_in_v, h,
    Nx::Int, Ny::Int,
    hmin::FT,
)
    """
    Reconstruct layer thickness `h` at v-points by averaging in y:

        h_in_v(i,j) = 0.5 * [h(i,j) + h(i,j+1)]

    - Clamped in y via `clamp1`.
    - Enforces a minimum thickness `hmin` at v-points.
    """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        jp1 = clamp1(j + 1, Ny)
        hf  = FT(0.5) * (h[i, j] + h[i, jp1])
        h_in_v[i, j] = max(hf, hmin)
    end
    return
end


# ============================================================
# Velocity reconstructions
# ============================================================

function k_recon_u_in_v!(
    u_in_v, u,
    Nx::Int, Ny::Int,
)
    """
    Reconstruct zonal velocity `u` from u-points to v-points
    using a 4-point area-average:

        u_in_v(i,j) = 0.25 * [u(i,j) + u(i,j+1) + u(i-1,j) + u(i-1,j+1)]

    - Periodic in x via `iper`
    - Clamped in y via `clamp1`
    """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        im1 = iper(i - 1, Nx)
        jp1 = clamp1(j + 1, Ny)
        u_in_v[i, j] = FT(0.25) * (
            u[i,  j]   + u[i,  jp1] +
            u[im1, j]  + u[im1, jp1]
        )
    end
    return
end

function k_recon_v_in_u!(
    v_in_u, v,
    Nx::Int, Ny::Int,
)
    """
    Reconstruct meridional velocity `v` from v-points to u-points
    using a 4-point area-average:

        v_in_u(i,j) = 0.25 * [v(i,j) + v(i+1,j) + v(i,j-1) + v(i+1,j-1)]

    - Periodic in x via `iper`
    - Clamped in y via `clamp1`
    """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        ip1 = iper(i + 1, Nx)
        jm1 = clamp1(j - 1, Ny)
        v_in_u[i, j] = FT(0.25) * (
            v[i,  j]   + v[ip1, j] +
            v[i,  jm1] + v[ip1, jm1]
        )
    end
    return
end


# ============================================================
# Face velocities for u-cells
# ============================================================

function k_calc_faceVels_for_ucell!(
    u_face_for_u, v_face_for_u,   # east & north face velocities of u-cell
    m, n,                         # transports at native u/v locations (m = h u, n = h v)
    h,                            # layer thickness at h-points
    Nx::Int, Ny::Int,
    hmin::FT,
)
    """
    Compute face-normal velocities for a **u-cell** (the transport cell for `m`).

    East face (u-normal):
        m_east = 0.5 * [ m(i,j) + m(i+1,j) ]
        h_east = h(i+1,j)
        u_face_for_u(i,j) = m_east / max(h_east, hmin)

    North face (v-normal):
        n_north = 0.5 * [ n(i,j) + n(i+1,j) ]
        h_north = 0.25 * [ h(i,j) + h(i+1,j) + h(i,j+1) + h(i+1,j+1) ]
        v_face_for_u(i,j) = n_north / max(h_north, hmin)

    - Periodic in x via `iper`
    - Clamped in y via `clamp1`
    - Divisions are guarded by `hmin`
    """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        ip1 = iper(i + 1, Nx)
        jp1 = clamp1(j + 1, Ny)

        # East face (u)
        m_east = FT(0.5) * (m[i, j] + m[ip1, j])
        h_east = max(h[ip1, j], hmin)
        u_face_for_u[i, j] = m_east / h_east

        # North face (v)
        n_north = FT(0.5) * (n[i, j] + n[ip1, j])
        h_north = FT(0.25) * (h[i, j] + h[ip1, j] + h[i, jp1] + h[ip1, jp1])
        h_north = max(h_north, hmin)
        v_face_for_u[i, j] = n_north / h_north
    end
    return
end


# ============================================================
# Face velocities for v-cells
# ============================================================

function k_calc_faceVels_for_vcell!(
    u_face_for_v, v_face_for_v,   # east & north face velocities of v-cell
    m, n,                         # transports at native u/v locations
    h,                            # layer thickness at h-points
    Nx::Int, Ny::Int,
    hmin::FT,
)
    """
    Compute face-normal velocities for a **v-cell** (the transport cell for `n`).

    East face (u-normal):
        m_east = 0.5 * [ m(i,j) + m(i,j+1) ]
        h_east = 0.25 * [ h(i,j) + h(i+1,j) + h(i,j+1) + h(i+1,j+1) ]
        u_face_for_v(i,j) = m_east / max(h_east, hmin)

    North face (v-normal):
        n_north = 0.5 * [ n(i,j) + n(i,j+1) ]
        h_north = h(i,j+1)
        v_face_for_v(i,j) = n_north / max(h_north, hmin)

    - Periodic in x via `iper`
    - Clamped in y via `clamp1`
    - Divisions are guarded by `hmin`
    """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        ip1 = iper(i + 1, Nx)
        jp1 = clamp1(j + 1, Ny)

        # East face (u)
        m_east = FT(0.5) * (m[i, j] + m[i, jp1])
        h_east = FT(0.25) * (h[i, j] + h[ip1, j] + h[i, jp1] + h[ip1, jp1])
        h_east = max(h_east, hmin)
        u_face_for_v[i, j] = m_east / h_east

        # North face (v)
        n_north = FT(0.5) * (n[i, j] + n[i, jp1])
        h_north = max(h[i, jp1], hmin)
        v_face_for_v[i, j] = n_north / h_north
    end
    return
end


# ============================================================
# Mode correction (GPU kernel + wrapper)
# ============================================================

function k_mode_correction!(
    h1, h2, H,
    m1, m2, M,
    n1, n2, N,
    w1::FT, w2::FT,
    hmin::FT,
    Nx::Int, Ny::Int,
    mode_split::Bool,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # out of bounds: do nothing
    if i > Nx || j > Ny
        return
    end

    # No mode splitting: keep barotropic fields consistent with layer sums.
    # (No correction/projection step.)
    if !mode_split
        H[i, j] = h1[i, j] + h2[i, j]
        M[i, j] = m1[i, j] + m2[i, j]
        N[i, j] = n1[i, j] + n2[i, j]
        return
    end

    # -------------------------------------------
    # Thickness correction: enforce H ≈ h1 + h2
    # -------------------------------------------
    ΔH = H[i, j] - h1[i, j] - h2[i, j]

    h1_new = h1[i, j] + w1 * ΔH
    h2_new = h2[i, j] + w2 * ΔH

    h1_new = max(h1_new, hmin)
    h2_new = max(h2_new, hmin)

    h1[i, j] = h1_new
    h2[i, j] = h2_new

    # -------------------------------------------
    # Transport correction: enforce M ≈ m1 + m2
    # -------------------------------------------
    Hsafe = max(H[i, j], hmin)

    ΔM = M[i, j] - m1[i, j] - m2[i, j]
    δU = ΔM / Hsafe

    m1[i, j] = m1[i, j] + h1_new * δU
    m2[i, j] = m2[i, j] + h2_new * δU

    # -------------------------------------------
    # Transport correction: enforce N ≈ n1 + n2
    # -------------------------------------------
    ΔN = N[i, j] - n1[i, j] - n2[i, j]
    δV = ΔN / Hsafe

    n1[i, j] = n1[i, j] + h1_new * δV
    n2[i, j] = n2[i, j] + h2_new * δV

    return
end

"""
    mode_correction!(state::State, p::Params;
                     threads::NTuple{2,Int},
                     blocks::NTuple{2,Int},
                     mode_split::Bool = true)

High-level GPU mode correction enforcing consistency between
barotropic and layer-wise variables when `mode_split == true`:

    H ≈ h1 + h2
    M ≈ m1 + m2
    N ≈ n1 + n2

Uses rest-depth weights

    w1 = H1 / (H1 + H2)
    w2 = H2 / (H1 + H2)

and calls `k_mode_correction!` on the GPU. If `mode_split == false`,
this routine does nothing.
"""
function mode_correction!(
    state::State,
    p::Params;
    threads::NTuple{2,Int},
    blocks::NTuple{2,Int},
    mode_split::Bool = true,
)
    prog = state.prog
    Nx, Ny = p.Nx, p.Ny

    # Rest-depth weights
    H1 = FT(p.H1)
    H2 = FT(p.H2)
    w1 = H1 / (H1 + H2)
    w2 = H2 / (H1 + H2)

    hmin = FT(p.hmin)

    @cuda threads=threads blocks=blocks k_mode_correction!(
        prog.h1, prog.h2, prog.H,
        prog.m1, prog.m2, prog.M,
        prog.n1, prog.n2, prog.N,
        w1, w2,
        hmin,
        Nx, Ny,
        mode_split,
    )

    return nothing
end
