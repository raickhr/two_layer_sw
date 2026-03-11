############################################################
# reconstruct.jl
#
# GPU kernels for C-grid reconstructions, face-velocity
# calculations, and barotropic–baroclinic consistency
# projection.
#----------------------------------------------------
# Reconstruction stencil (C-grid cell), conceptually:
#
#    ζ(i-1,j)------------- v(i,j) -------------ζ(i, j)
#        |                                       |
#        |                                       |
#        |                                       |
#     u(i-1,j)             h(i,j)              u(i,j)
#        |                                       |
#        |                                       |
#        |                                       |
#  ζ(i-1,j-1)------------- v(i,j-1) ------------ζ(i,j-1)
#
# Here, h_in_u and h_in_v already live at u- and v-
# points; we just sample them with stencil(..., 0,0,...).
#----------------------------------------------------
#
# Assumptions:
#   - FT, State, Params, stencil, HGRID, UGRID, VGRID
#     are defined in module scope
#   - CUDA is available
#
# All kernels are allocation-free and GPU-safe.
############################################################


# ============================================================
# Thickness reconstructions
# ============================================================

function k_recon_h_in_u!(
    h_in_u, h,
    Nx::Int, Ny::Int,
)
    # h at u-points via x-average:
    # h_in_u(i,j) = 0.5 [h(i,j) + h(i+1,j)]
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        hL = stencil(h, i, j,  0, 0, Nx, Ny, HGRID)
        hR = stencil(h, i, j, +1, 0, Nx, Ny, HGRID)
        h_in_u[i, j] = FT(0.5)*(hL + hR)
    end
    return
end


function k_recon_h_in_v!(
    h_in_v, h,
    Nx::Int, Ny::Int,
)
    # h at v-points via y-average:
    # h_in_v(i,j) = 0.5 [h(i,j) + h(i,j+1)]
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        hS = stencil(h, i, j, 0,  0, Nx, Ny, HGRID)
        hN = stencil(h, i, j, 0, +1, Nx, Ny, HGRID)
        h_in_v[i, j] = FT(0.5)*(hS + hN)
    end
    return
end


# ============================================================
# Velocity reconstructions (cross-stagger)
# ============================================================

function k_recon_u_in_v!(
    u_in_v, u,
    Nx::Int, Ny::Int,
)
    # u at v-points via 4-point average
    # u_in_v(i,j) = 0.25 [u(i,j) + u(i,j+1) + u(i-1,j) + u(i-1,j+1)]
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        u_in_v[i, j] = FT(0.25)*(
            stencil(u,i,j,  0, 0,Nx,Ny,UGRID) +
            stencil(u,i,j,  0,+1,Nx,Ny,UGRID) +
            stencil(u,i,j, -1, 0,Nx,Ny,UGRID) +
            stencil(u,i,j, -1,+1,Nx,Ny,UGRID)
        )
    end
    return
end


function k_recon_v_in_u!(
    v_in_u, v,
    Nx::Int, Ny::Int,
)
    # v at u-points via 4-point average
    # v_in_u(i,j) = 0.25 [v(i,j) + v(i+1,j) + v(i,j-1) + v(i+1,j-1)]
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        v_in_u[i, j] = FT(0.25)*(
            stencil(v, i, j,  0,  0, Nx, Ny, VGRID) +
            stencil(v, i, j, +1,  0, Nx, Ny, VGRID) +
            stencil(v, i, j,  0, -1, Nx, Ny, VGRID) +
            stencil(v, i, j, +1, -1, Nx, Ny, VGRID)
        )
    end
    return
end

##########################
#
##########################


function k_recon_z_in_v!(
    z_in_v, z,
    Nx::Int, Ny::Int
)
    # h at v-points via y-average:
    # h_in_v(i,j) = 0.5 [h(i,j) + h(i,j+1)]
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        zW = stencil(z, i, j, -1, 0, Nx, Ny, ZGRID)
        zE = stencil(z, i, j,  0, 0, Nx, Ny, ZGRID)
        z_in_v[i, j] = FT(0.5)*(zW + zE)
    end
    return
end

function k_recon_z_in_u!(
    z_in_u, z,
    Nx::Int, Ny::Int
)
    # h at v-points via y-average:
    # h_in_v(i,j) = 0.5 [h(i,j) + h(i,j+1)]
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        zS = stencil(z, i, j, 0,-1, Nx, Ny, ZGRID)
        zN = stencil(z, i, j, 0, 0, Nx, Ny, ZGRID)
        z_in_u[i, j] = FT(0.5)*(zS + zN)
    end
    return
end


# ============================================================
# Face velocities for u-cells (advecting m)
# ============================================================

function k_calc_faceVels_for_ucell!(
    u_face, v_face,
    m, n,
    h,
    Nx::Int, Ny::Int,
    hmin::FT,
)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny

        # East face (u-normal)
        m_e = FT(0.5)*(
            stencil(m,i,j, 0,0,Nx,Ny,UGRID) +
            stencil(m,i,j,+1,0,Nx,Ny,UGRID)
        )

        h_e = max(stencil(h,i,j,+1,0,Nx,Ny,HGRID), hmin)
        u_face[i,j] = m_e / h_e

        # North face (v-normal)
        n_n = FT(0.5)*(
            stencil(n,i,j, 0,0,Nx,Ny,VGRID) +
            stencil(n,i,j,+1,0,Nx,Ny,VGRID)
        )

        h_n = FT(0.25)*(
            stencil(h,i,j, 0, 0,Nx,Ny,HGRID) +
            stencil(h,i,j,+1, 0,Nx,Ny,HGRID) +
            stencil(h,i,j, 0,+1,Nx,Ny,HGRID) +
            stencil(h,i,j,+1,+1,Nx,Ny,HGRID)
        )

        v_face[i,j] = n_n / max(h_n,hmin)
    end
    return
end


# ============================================================
# Face velocities for v-cells (advecting n)
# ============================================================

function k_calc_faceVels_for_vcell!(
    u_face, v_face,
    m, n,
    h,
    Nx::Int, Ny::Int,
    hmin::FT,
)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny

        # East face (u-normal)
        m_e = FT(0.5)*(
            stencil(m,i,j, 0,0,Nx,Ny,UGRID) +
            stencil(m,i,j, 0,+1,Nx,Ny,UGRID)
        )

        h_e = FT(0.25)*(
            stencil(h,i,j, 0,0,Nx,Ny,HGRID) +
            stencil(h,i,j,+1,0,Nx,Ny,HGRID) +
            stencil(h,i,j, 0,+1,Nx,Ny,HGRID) +
            stencil(h,i,j,+1,+1,Nx,Ny,HGRID)
        )

        u_face[i,j] = m_e / max(h_e,hmin)

        # North face (v-normal)
        # Between v(i,j) and v(i,j+1)
        n_n = FT(0.5)*(
            stencil(n,i,j, 0,0,Nx,Ny,VGRID) +
            stencil(n,i,j, 0,+1,Nx,Ny,VGRID)
        )

        h_n = max(stencil(h,i,j,0,+1,Nx,Ny,HGRID), hmin)
        v_face[i,j] = n_n / h_n
    end
    return
end


# ============================================================
# Mode correction (barotropic consistency projection)
# ============================================================

function k_mode_correction!(
    h1, h2, H,
    m1, m2, M,
    n1, n2, N,
    hmin::FT,
    Nx::Int, Ny::Int,
    mode_split::Bool,
)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i > Nx || j > Ny
        return
    end

    if !mode_split
        H[i,j] = h1[i,j] + h2[i,j]
        M[i,j] = m1[i,j] + m2[i,j]
        N[i,j] = n1[i,j] + n2[i,j]
        return
    end

    # Treat (H,M,N) as reference barotropic fields.
    denom = max(h1[i,j] + h2[i,j], hmin)
    w1 = h1[i,j] / denom
    w2 = h2[i,j] / denom

    # Thickness projection
    ΔH = H[i,j] - h1[i,j] - h2[i,j]

    h1[i,j] = max(h1[i,j] + w1*ΔH, hmin)
    h2[i,j] = max(h2[i,j] + w2*ΔH, hmin)

    # Transport projection
    ΔM = M[i,j] - m1[i,j] - m2[i,j]
    m1[i,j] += w1*ΔM
    m2[i,j] += w2*ΔM

    ΔN = N[i,j] - n1[i,j] - n2[i,j]
    n1[i,j] += w1*ΔN
    n2[i,j] += w2*ΔN

    return
end


function mode_correction!(
    state::State,
    p::Params;
    threads::NTuple{2,Int},
    blocks::NTuple{2,Int},
    mode_split::Bool = true,
)

    prog = state.prog
    Nx = Int(p.Nx)
    Ny = Int(p.Ny)
    hmin = FT(p.hmin)

    @cuda threads=threads blocks=blocks k_mode_correction!(
        prog1.h, prog2.h, prog.H,
        prog1.m, prog2.m, prog.M,
        prog1.n, prog2.n, prog.N,
        hmin,
        Nx, Ny,
        mode_split,
    )

    return nothing
end