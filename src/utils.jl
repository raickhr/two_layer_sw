############################################################
# utils_diffusion.jl
#
# Utility functions and GPU kernels for the rotating
# shallow-water C-grid model.
#
# GRID / STAGGERING CONVENTION
# ----------------------------
# All fields have array size (Nx, Ny), but their physical
# meaning in the meridional (j) direction depends on the
# grid type and ghost rows:
#
#   - HGRID (cell centers, "h-points"):
#       j = 1     : ghost row (just south of the south boundary)
#       j = 2..Ny : physical interior up to just inside v(j=Ny)
#
#   - UGRID (zonal faces, "u-points"):
#       j = 1     : ghost row (just south of the south boundary)
#       j = 2..Ny : physical interior
#
#   - VGRID (meridional faces, "v-points"):
#       j = 1     : south boundary (on the wall)
#       j = Ny    : north boundary (on the wall)
#
#   - ZGRID (corners, "zeta-points"):
#       j = 1..Ny : span the domain in y
#
# The device-side helper `stencil(field, i, j, di, dj, Nx, Ny, grid)`
# enforces this layout:
#
#   * periodic wrap in x for all grids (via iper),
#   * in y:
#       - HGRID / UGRID → clamp2(j,N) ∈ [2,Ny]
#       - VGRID / ZGRID → clamp1(j,N) ∈ [1,Ny]
#
# Solid walls and ghost values themselves are enforced by the
# BC kernels in `boundaries.jl`; `stencil` only keeps indices
# consistent with the staggering + ghost-row design.
#
# Provides:
#   - coriolis_at_lat                    : Coriolis parameter f = 2Ω sin(lat)
#   - calcCourant                        : Courant number using FV geometry
#   - k_calc_gradient!                   : center → east/north-face gradients
#   - k_calc_gradient_u!                 : gradients of u (UGRID) to H/Z points
#   - k_calc_div_of_gradient_u!          : divergence of grad(u) back to UGRID
#   - k_calc_gradient_v!                 : gradients of v (VGRID) to Z/H points
#   - k_calc_div_of_gradient_v!          : divergence of grad(v) back to VGRID
#   - k_calc_gradient_z!                 : gradients of ZGRID field to U/V
#   - k_calc_viscous_from_gradu_in_ucell!: ∇·(h ν ∇u) on UGRID
#   - k_calc_viscous_from_gradv_in_vcell!: ∇·(h ν ∇v) on VGRID
#   - calculate_leith_viscosity!         : ν ∝ Δ³ |∇ζ| on U/V grids
#   - calculate_viscous_term!            : apply ∇·(h ν ∇u), ∇·(h ν ∇v) for given ν
#   - k_calc_flux_m_for_ucell!           : centered advective flux of m = h u
#   - k_calc_flux_n_for_vcell!           : centered advective flux of n = h v
#
# Assumptions:
#   - Scalars at cell centers (HGRID); u on zonal faces (UGRID);
#     v on meridional faces (VGRID); z at corners (ZGRID).
#   - `stencil(field, i, j, di, dj, Nx, Ny, grid)` implements
#     periodicity in x and stagger-aware clamping in y as
#     described above.
#   - All functions are CUDA-GPU safe (no allocations, simple arithmetic).
#
# All gradient/divergence operators follow standard flux-form FV
# conventions on this staggered C-grid.
############################################################


# ---------------------------------------------------------
# Simple scalar helpers
# ---------------------------------------------------------

@inline coriolis_at_lat(Ω::FT, latInDeg::FT) =
    FT(2) * Ω * sin(latInDeg * (FT(pi) / FT(180)))

@inline calcCourant(vel::FT, dy::FT, dArea::FT, dt::FT) =
    vel * dy / dArea * dt


# ---------------------------------------------------------
# Cell-center gradient to east/north faces
# ---------------------------------------------------------

"""
    k_calc_gradient!(gradx_face, grady_face,
                     phi, dx_n2n, dy_n2n,
                     Nx, Ny)

Compute forward gradients from cell centers (HGRID) to east and
north faces:

    gradx_face[i,j] ≈ (phi[i+1,j] - phi[i,j]) / dx_n2n[i,j]
    grady_face[i,j] ≈ (phi[i,j+1] - phi[i,j]) / dy_n2n[i,j]

Indices and periodic/ghost behavior are handled by `stencil`.
"""
function k_calc_gradient_h!(
    gradx_face, grady_face,
    phi, dx_n2n, dy_n2n,
    Nx::Int, Ny::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        gradx_face[i, j] =
            (stencil(phi, i, j, +1, 0, Nx, Ny, HGRID) -
             stencil(phi, i, j,  0, 0, Nx, Ny, HGRID)) /
             stencil(dx_n2n, i, j, 0, 0, Nx, Ny, GRID0)

        grady_face[i, j] =
            (stencil(phi, i, j, 0, +1, Nx, Ny, HGRID) -
             stencil(phi, i, j, 0,  0, Nx, Ny, HGRID)) /
             stencil(dy_n2n, i, j, 0, 0, Nx, Ny, GRID0)
    end
    return
end


# ---------------------------------------------------------
# Grad / div-grad for u-points (UGRID)
# ---------------------------------------------------------

"""
    k_calc_gradient_u!(gradx_in_h, grady_in_z,
                       phi_in_u, dx_n2n_u, dy_n2n_u,
                       Nx, Ny)

Compute gradients of a UGRID field `phi_in_u`:

- ∂φ/∂x at HGRID (cell centers), backward difference in x;
- ∂φ/∂y at ZGRID (corners), forward difference in y.
"""
function k_calc_gradient_u!(
    gradx_in_h, grady_in_z,
    phi_in_u, dx_n2n_u, dy_n2n_u,
    Nx::Int, Ny::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        # ∂φ/∂x at h-points (HGRID): centered between u[i-1] and u[i]
        gradx_in_h[i, j] =
            (stencil(phi_in_u, i, j,  0, 0, Nx, Ny, UGRID) -
             stencil(phi_in_u, i, j, -1, 0, Nx, Ny, UGRID)) /
             stencil(dx_n2n_u, i, j, -1, 0, Nx, Ny, GRID0)

        # ∂φ/∂y at z-points (ZGRID): forward in j
        grady_in_z[i, j] =
            (stencil(phi_in_u, i, j,  0, +1, Nx, Ny, UGRID) -
             stencil(phi_in_u, i, j,  0,  0, Nx, Ny, UGRID)) /
             stencil(dy_n2n_u, i, j, 0, 0, Nx, Ny, GRID0)
    end
    return
end

"""
    k_calc_div_of_gradient_u!(div,
                              gradx_in_h, grady_in_z,
                              dx_n2n_h, dy_n2n_z,
                              Nx, Ny)

Divergence of gradients of a UGRID field:

    div ≈ ∂/∂x(gradx_in_h) + ∂/∂y(grady_in_z)

where `gradx_in_h` is on HGRID and `grady_in_z` on ZGRID.
"""
function k_calc_div_of_gradient_u!(
    div,
    gradx_in_h, grady_in_z,
    dx_n2n_h, dy_n2n_z,
    Nx::Int, Ny::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        # ∂/∂x (gradx at HGRID) → UGRID
        xterm =
            (stencil(gradx_in_h, i, j,  +1, 0, Nx, Ny, HGRID) -
             stencil(gradx_in_h, i, j,   0, 0, Nx, Ny, HGRID)) /
            stencil(dx_n2n_h, i, j, 0, 0, Nx, Ny, GRID0)

        # ∂/∂y (grady at ZGRID) → UGRID
        yterm =
            (stencil(grady_in_z, i, j, 0, 0,  Nx, Ny, ZGRID) -
             stencil(grady_in_z, i, j, 0, -1, Nx, Ny, ZGRID)) /
            stencil(dy_n2n_z, i, j, 0, -1, Nx, Ny, GRID0)

        div[i, j] = xterm + yterm
    end
    return
end


# ---------------------------------------------------------
# Grad / div-grad for v-points (VGRID)
# ---------------------------------------------------------

"""
    k_calc_gradient_v!(gradx_in_z, grady_in_h,
                       phi_in_v, dx_n2n_v, dy_n2n_v,
                       Nx, Ny)

Compute gradients of a VGRID field `phi_in_v`:

- ∂φ/∂x at ZGRID (corners), forward difference in x;
- ∂φ/∂y at HGRID (cell centers), backward difference in y.
"""
function k_calc_gradient_v!(
    gradx_in_z, grady_in_h,
    phi_in_v, dx_n2n_v, dy_n2n_v,
    Nx::Int, Ny::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        # ∂φ/∂x at ZGRID
        gradx_in_z[i, j] =
            (stencil(phi_in_v, i, j,  +1, 0, Nx, Ny, VGRID) -
             stencil(phi_in_v, i, j,   0, 0, Nx, Ny, VGRID)) /
             stencil(dx_n2n_v, i, j,  0, 0, Nx, Ny, GRID0)

        # ∂φ/∂y at HGRID
        grady_in_h[i, j] =
            (stencil(phi_in_v, i, j,  0,  0, Nx, Ny, VGRID) -
             stencil(phi_in_v, i, j,  0, -1, Nx, Ny, VGRID)) /
             stencil(dy_n2n_v, i, j,  0, -1, Nx, Ny, GRID0)
    end
    return
end

"""
    k_calc_div_of_gradient_v!(div,
                              gradx_in_z, grady_in_h,
                              dx_n2n_z, dy_n2n_h,
                              Nx, Ny)

Divergence of gradients of a VGRID field:

    div ≈ ∂/∂x(gradx_in_z) + ∂/∂y(grady_in_h)

where `gradx_in_z` is on ZGRID and `grady_in_h` on HGRID.
"""
function k_calc_div_of_gradient_v!(
    div,
    gradx_in_z, grady_in_h,
    dx_n2n_z, dy_n2n_h,
    Nx::Int, Ny::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        # ∂/∂x (gradx at ZGRID) → VGRID
        xterm =
            (stencil(gradx_in_z, i, j,  0, 0, Nx, Ny, ZGRID) -
             stencil(gradx_in_z, i, j, -1, 0, Nx, Ny, ZGRID)) /
            stencil(dx_n2n_z, i, j, -1, 0, Nx, Ny, GRID0)
        
        # ∂/∂y (grady at HGRID) → VGRID
        yterm =
            (stencil(grady_in_h, i, j,   0, +1, Nx, Ny, HGRID) -
             stencil(grady_in_h, i, j,   0,  0, Nx, Ny, HGRID)) /
            stencil(dy_n2n_h, i, j,  0, 0, Nx, Ny, GRID0)

        div[i, j] = xterm + yterm
    end
    return
end


# ---------------------------------------------------------
# Gradients of ZGRID field (for Leith viscosity)
# ---------------------------------------------------------

"""
    k_calc_gradient_z!(gradx_in_v, grady_in_u,
                       phi_in_z, dx_n2n_z, dy_n2n_z,
                       Nx, Ny)

Compute ∂φ/∂x and ∂φ/∂y of a ZGRID field `phi_in_z`, returning
∂φ/∂x on VGRID (`gradx_in_v`) and ∂φ/∂y on UGRID (`grady_in_u`)
using backward differences and stagger-aware indexing.
"""
function k_calc_gradient_z!(
    gradx_in_v, grady_in_u,
    phi_in_z, dx_n2n_z, dy_n2n_z,
    Nx::Int, Ny::Int,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        # ∂φ/∂x using backward difference at ZGRID → interpreted on VGRID
        gradx_in_v[i, j] =
            (stencil(phi_in_z, i, j,  0, 0, Nx, Ny, ZGRID) -
             stencil(phi_in_z, i, j, -1, 0, Nx, Ny, ZGRID)) /
            stencil(dx_n2n_z, i, j, -1, 0, Nx, Ny, GRID0)

        # ∂φ/∂y using backward difference at ZGRID → interpreted on UGRID
        grady_in_u[i, j] =
            (stencil(phi_in_z, i, j, 0,  0, Nx, Ny, ZGRID) -
             stencil(phi_in_z, i, j, 0, -1, Nx, Ny, ZGRID)) /
            stencil(dy_n2n_z, i, j, 0, -1, Nx, Ny, GRID0)
    end
    return
end


# ---------------------------------------------------------
# Flux-form Leith diffusion from ∇u, ∇v
# ---------------------------------------------------------

"""
    k_calc_viscous_from_gradu_in_ucell!(
        viscous_m,
        gradx_u_in_h, grady_u_in_z,
        h, ν_leith_in_u,
        dx_face_u, dy_face_u, dArea_u,
        Nx, Ny, hmin,
    )

Compute the flux-form diffusion of zonal velocity u on UGRID:

    viscous_m ≈ ∇ · ( h ν ∇u )

using precomputed ∂u/∂x on HGRID and ∂u/∂y on ZGRID, and metric
(face-length and area) information on the u-grid.
"""
function k_calc_viscous_from_gradu_in_ucell!(
    viscous_m,                      # ∇⋅(h ν ∇u) at u-cells
    gradx_u_in_h,  grady_u_in_z,    # ∂x(u) on HGRID, ∂y(u) on ZGRID
    h,                              # layer thickness at HGRID
    ν_leith_in_u,                   # viscosity at UGRID
    dx_face_u, dy_face_u,           # face lengths around UGRID cells
    dArea_u,                        # u-cell area
    Nx::Int, Ny::Int,
    hmin::FT,
)
    # Global thread indices
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        # =====================================================
        # Zonal faces (east / west) of the u-cell (i,j)
        # =====================================================

        # East face: between u(i,j) and u(i+1,j)
        dxu_east = stencil(gradx_u_in_h, i, j,  1, 0, Nx, Ny, HGRID)  # ∂u/∂x at east face
        h_east   = max(stencil(h, i, j,  1, 0, Nx, Ny, HGRID), hmin)
        ν_east   = FT(0.5) * (
            abs(stencil(ν_leith_in_u, i, j,  0, 0, Nx, Ny, UGRID)) +
            abs(stencil(ν_leith_in_u, i, j,  1, 0, Nx, Ny, UGRID))
        )
        dy_east  = stencil(dy_face_u, i, j,  0, 0, Nx, Ny, GRID0)

        # West face: between u(i-1,j) and u(i,j)
        dxu_west = stencil(gradx_u_in_h, i, j,  0, 0, Nx, Ny, HGRID)  # ∂u/∂x at west face
        h_west   = max(stencil(h, i, j,  0, 0, Nx, Ny, HGRID), hmin)
        ν_west   = FT(0.5) * (
            abs(stencil(ν_leith_in_u, i, j, -1, 0, Nx, Ny, UGRID)) +
            abs(stencil(ν_leith_in_u, i, j,  0, 0, Nx, Ny, UGRID))
        )
        dy_west  = stencil(dy_face_u, i, j, -1, 0, Nx, Ny, GRID0)

        # =====================================================
        # Meridional faces (north / south) of the u-cell (i,j)
        # =====================================================

        # North face: between rows j and j+1
        dyu_north = stencil(grady_u_in_z, i, j,  0, 0, Nx, Ny, ZGRID)  # ∂u/∂y at north face
        h_north   = FT(0.25) * (
            stencil(h, i, j,  0, 0, Nx, Ny, HGRID) +
            stencil(h, i, j,  1, 0, Nx, Ny, HGRID) +
            stencil(h, i, j,  1, 1, Nx, Ny, HGRID) +
            stencil(h, i, j,  0, 1, Nx, Ny, HGRID)
        )
        h_north   = max(h_north, hmin)
        ν_north   = FT(0.5) * (
            abs(stencil(ν_leith_in_u, i, j,  0, 0, Nx, Ny, UGRID)) +
            abs(stencil(ν_leith_in_u, i, j,  0, 1, Nx, Ny, UGRID))
        )
        dx_north  = stencil(dx_face_u, i, j,  0, 0, Nx, Ny, GRID0)

        # South face: between rows j-1 and j
        dyu_south = stencil(grady_u_in_z, i, j,  0, -1, Nx, Ny, ZGRID) # ∂u/∂y at south face
        h_south   = FT(0.25) * (
            stencil(h, i, j,  0,  0, Nx, Ny, HGRID) +
            stencil(h, i, j,  1,  0, Nx, Ny, HGRID) +
            stencil(h, i, j,  0, -1, Nx, Ny, HGRID) +
            stencil(h, i, j,  1, -1, Nx, Ny, HGRID)
        )
        h_south   = max(h_south, hmin)
        ν_south   = FT(0.5) * (
            abs(stencil(ν_leith_in_u, i, j, 0, -1, Nx, Ny, UGRID)) +
            abs(stencil(ν_leith_in_u, i, j, 0,  0, Nx, Ny, UGRID))
        )
        dx_south  = stencil(dx_face_u, i, j,  0, -1, Nx, Ny, GRID0)

        # Cell area at this u-point
        areaU = stencil(dArea_u, i, j, 0, 0, Nx, Ny, GRID0)

        # =====================================================
        # Flux divergence: ∇·(h ν ∇u) / areaU
        # =====================================================
        # Zonal fluxes: F_w, F_e = (h ν ∂u/∂x) * (meridional face length)
        x_comp = (dy_east * dxu_east * h_east * ν_east -
                  dy_west * dxu_west * h_west * ν_west) / areaU

        # Meridional fluxes: F_s, F_n = (h ν ∂u/∂y) * (zonal face length)
        y_comp = (dx_north * dyu_north * h_north * ν_north -
                  dx_south * dyu_south * h_south * ν_south ) / areaU


        # # Zonal fluxes: F_w, F_e = (h ν ∂u/∂x) * (meridional face length)
        # h_in_u = FT(0.5) * (stencil(h, i, j, 1, 0, Nx, Ny, GRID0) +
        #                     stencil(h, i, j, 0, 0, Nx, Ny, GRID0))
        # x_comp = h_in_u * (dy_east * dxu_east * ν_east -
        #           dy_west * dxu_west * ν_west) / areaU

        # # Meridional fluxes: F_s, F_n = (h ν ∂u/∂y) * (zonal face length)
        # y_comp = h_in_u *(dx_north * dyu_north * ν_north -
        #           dx_south * dyu_south * ν_south ) / areaU

        # Total divergence at u-cell (i,j)
        viscous_m[i, j] = x_comp + y_comp
    end

    return
end

"""
    k_calc_viscous_from_gradv_in_vcell!(
        viscous_n,
        gradx_v_in_z, grady_v_in_h,
        h, ν_leith_in_v,
        dx_face_v, dy_face_v, dArea_v,
        Nx, Ny, hmin,
    )

Compute the flux-form diffusion of meridional velocity `v`
on VGRID:

    viscous_n ≈ ∇ · ( h ν ∇v )

using precomputed ∂v/∂x on ZGRID and ∂v/∂y on HGRID, together with
face-length and area metrics on the v-grid.
"""
function k_calc_viscous_from_gradv_in_vcell!(
    viscous_n,                      # ∇⋅(h ν ∇v) at v-cells (VGRID)
    gradx_v_in_z,  grady_v_in_h,    # ∂x(v) on ZGRID, ∂y(v) on HGRID
    h,                              # layer thickness at HGRID
    ν_leith_in_v,                   # viscosity at VGRID
    dx_face_v, dy_face_v,           # face lengths around VGRID cells
    dArea_v,                        # v-cell area
    Nx::Int, Ny::Int,
    hmin::FT,
)
    # Global thread indices
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        # =====================================================
        # Zonal faces (east / west) of the v-cell (i,j)
        # =====================================================

        # East face: between v(i,j) and v(i+1,j)
        dxv_east = stencil(gradx_v_in_z, i, j,  0, 0, Nx, Ny, ZGRID)
        h_east   = FT(0.25) * (
            stencil(h, i, j,  0, 0, Nx, Ny, HGRID) +
            stencil(h, i, j,  0, 1, Nx, Ny, HGRID) +
            stencil(h, i, j,  1, 0, Nx, Ny, HGRID) +
            stencil(h, i, j,  1, 1, Nx, Ny, HGRID)
        )
        h_east   = max(h_east, hmin)
        ν_east   = FT(0.5) * (
            abs(stencil(ν_leith_in_v, i, j,  0, 0, Nx, Ny, VGRID)) +
            abs(stencil(ν_leith_in_v, i, j,  1, 0, Nx, Ny, VGRID))
        )
        dy_east  = stencil(dy_face_v, i, j,  0, 0, Nx, Ny, GRID0)

        # West face: between v(i-1,j) and v(i,j)
        dxv_west = stencil(gradx_v_in_z, i, j, -1, 0, Nx, Ny, ZGRID)
        h_west   = FT(0.25) * (
            stencil(h, i, j, -1, 0, Nx, Ny, HGRID) +
            stencil(h, i, j, -1, 1, Nx, Ny, HGRID) +
            stencil(h, i, j,  0, 0, Nx, Ny, HGRID) +
            stencil(h, i, j,  0, 1, Nx, Ny, HGRID)
        )
        h_west   = max(h_west, hmin)
        ν_west   = FT(0.5) * (
            abs(stencil(ν_leith_in_v, i, j, -1, 0, Nx, Ny, VGRID)) +
            abs(stencil(ν_leith_in_v, i, j,  0, 0, Nx, Ny, VGRID))
        )
        dy_west  = stencil(dy_face_v, i, j, -1, 0, Nx, Ny, GRID0)

        # =====================================================
        # Meridional faces (north / south) of the v-cell (i,j)
        # =====================================================

        # North face: between v(i,j) and v(i,j+1)
        dyv_north = stencil(grady_v_in_h, i, j,  0, 1, Nx, Ny, HGRID)
        h_north   = max(stencil(h, i, j,  0, 1, Nx, Ny, HGRID), hmin)
        ν_north   = FT(0.5) * (
            abs(stencil(ν_leith_in_v, i, j, 0, 0, Nx, Ny, VGRID)) +
            abs(stencil(ν_leith_in_v, i, j, 0, 1, Nx, Ny, VGRID))
        )
        dx_north  = stencil(dx_face_v, i, j,  0, 0, Nx, Ny, GRID0)

        # South face: between v(i,j-1) and v(i,j)
        dyv_south = stencil(grady_v_in_h, i, j,  0, 0, Nx, Ny, HGRID)
        h_south   = max(stencil(h, i, j,  0, 0, Nx, Ny, HGRID), hmin)
        ν_south   = FT(0.5) * (
            abs(stencil(ν_leith_in_v, i, j, 0, -1, Nx, Ny, VGRID)) +
            abs(stencil(ν_leith_in_v, i, j, 0,  0, Nx, Ny, VGRID))
        )
        dx_south  = stencil(dx_face_v, i, j,  0, -1, Nx, Ny, GRID0)

        # Cell area at this v-point
        areaV = stencil(dArea_v, i, j, 0, 0, Nx, Ny, GRID0)

        # =====================================================
        # Flux divergence: ∇·(h ν ∇v) / areaV
        # =====================================================
        # Zonal fluxes: F_w, F_e = (h ν ∂v/∂x) * (meridional face length)
        x_comp = (dy_east * dxv_east * h_east * ν_east -
                  dy_west * dxv_west * h_west * ν_west ) / areaV

        # Meridional fluxes: F_s, F_n = (h ν ∂v/∂y) * (zonal face length)
        y_comp = (dx_north * dyv_north * h_north * ν_north -
                  dx_south * dyv_south * h_south * ν_south) / areaV

        # # Zonal fluxes: F_w, F_e = (h ν ∂v/∂x) * (meridional face length)
        # h_in_v = FT(0.5) * (stencil(h, i, j, 0 , 0 , Nx, Ny, HGRID) + 
        #                     stencil(h, i, j, 0 , 1 , Nx, Ny, HGRID) )
        # x_comp = h_in_v  * (dy_east * dxv_east * ν_east -
        #                     dy_west * dxv_west * ν_west) / areaV

        # # Meridional fluxes: F_s, F_n = (h ν ∂v/∂y) * (zonal face length)
        # y_comp = h_in_v * (dx_north * dyv_north * ν_north -
        #                    dx_south * dyv_south * ν_south) / areaV

        # Total divergence at v-cell (i,j)
        viscous_n[i, j] = x_comp + y_comp
    end

    return
end


# ---------------------------------------------------------
# Leith-type viscosity: ν ∝ Δ³ |∇ζ|
# ---------------------------------------------------------
function calculate_leith_viscosity!(
    leithViscosity_in_u, 
    leithViscosity_in_v,
    zeta, 
    dx_n2n_z, dy_n2n_z,
    dx_n2n_u, dy_n2n_u,
    dx_n2n_v, dy_n2n_v,
    buf1_x, buf1_y,
    buf2_x, buf2_y,
    minimum_vis,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
    Nx::Int, Ny::Int, 
    C::FT = FT(0.8),
)
    ##############################
    # ∇ζ on UGRID / VGRID
    ##############################
    gradx_zeta_in_v = buf1_x
    grady_zeta_in_u = buf1_y

    @cuda threads=threads2 blocks=blocks2 k_calc_gradient_z!(
        gradx_zeta_in_v, grady_zeta_in_u,
        zeta, dx_n2n_z, dy_n2n_z,
        Nx, Ny,
    )

    gradx_zeta_in_u = buf2_x
    grady_zeta_in_v = buf2_y

    @cuda threads=threads2 blocks=blocks2 k_recon_v_in_u!(
        gradx_zeta_in_u, gradx_zeta_in_v, Nx, Ny,
    )
    @cuda threads=threads2 blocks=blocks2 k_recon_u_in_v!(
        grady_zeta_in_v, grady_zeta_in_u, Nx, Ny,
    )

    ##############################
    # Leith viscosity ν = C³ Δ³ |∇ζ|,  with Δ ≈ √(dx·dy)
    ##############################

    @. leithViscosity_in_u = C^3 *
                (sqrt(dx_n2n_u * dy_n2n_u))^3 *
                sqrt(gradx_zeta_in_u^2 + grady_zeta_in_u^2)

    @. leithViscosity_in_v = C^3 *
                (sqrt(dx_n2n_v * dy_n2n_v))^3 *
                sqrt(gradx_zeta_in_v^2 + grady_zeta_in_v^2)

    @. leithViscosity_in_u = max(leithViscosity_in_u, minimum_vis)
    @. leithViscosity_in_v = max(leithViscosity_in_v, minimum_vis)

end

function calculate_viscous_term!(
    viscous_m, viscous_n,
    viscosity_in_u, viscosity_in_v,
    h,
    gradx_u, grady_u,
    gradx_v, grady_v,
    dx_face_u, dy_face_u, dArea_u,   
    dx_face_v, dy_face_v, dArea_v,   
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
    Nx::Int, Ny::Int;
    hmin::FT = FT(1),
)
    ##############################
    # Apply ∇⋅(h ν ∇u) and ∇⋅(h ν ∇v)
    ##############################
    @cuda threads=threads2 blocks=blocks2 k_calc_viscous_from_gradu_in_ucell!(
        viscous_m,                      # ∇⋅(h ν ∇u) at u-cells
        gradx_u,  grady_u,              # ∂x(u) on HGRID, ∂y(u) on ZGRID
        h,                              # layer thickness at HGRID
        viscosity_in_u,                 # viscosity at UGRID
        dx_face_u, dy_face_u,           # face lengths around UGRID cells
        dArea_u,                        # u-cell area
        Nx, Ny,
        hmin,
    )

    @cuda threads=threads2 blocks=blocks2 k_calc_viscous_from_gradv_in_vcell!(
        viscous_n,                      # ∇⋅(h ν ∇v) at v-cells
        gradx_v,  grady_v,              # ∂x(v) on ZGRID, ∂y(v) on HGRID
        h,                              # layer thickness at HGRID
        viscosity_in_v,                 # viscosity at VGRID
        dx_face_v, dy_face_v,           # face lengths around VGRID cells
        dArea_v,                        # v-cell area
        Nx, Ny,
        hmin,
    )

    return
end


# ---------------------------------------------------------
# Centered conservative momentum-flux divergence
# for m = h*u on u-cells (UGRID)
# ---------------------------------------------------------

"""
    k_calc_flux_m_for_ucell!(m_flux, m, n, h,
                             dx_face_u, dy_face_u, dArea_u,
                             Nx, Ny, g, hmin)

Compute the centered finite-volume conservative flux-divergence term
for zonal momentum `m = h*u` on C-grid u-cells (UGRID), including
metric factors and the hydrostatic pressure flux.

This kernel evaluates the discrete divergence of the x-momentum flux
tensor in conservative form:

    ∂x(hu² + 1/2 g h²) + ∂y(huv)

Using finite-volume notation, the returned quantity is

    m_flux[i,j] = (F_west - F_east + F_south - F_north) / areaU

where

- `F_x = (hu² + 1/2 g h²) * dy`  is the zonal face flux
- `F_y = (huv) * dx`             is the meridional face flux

Notes
-----
- `m` is stored on UGRID and represents `h*u`.
- `n` is stored on VGRID and represents `h*v`.
- `h` is stored on HGRID.
- Face/corner values are obtained by centered interpolation.
- Thickness used in velocity reconstruction is clipped below by `hmin`
  to avoid division by very small values.
"""
function k_calc_flux_m_for_ucell!(
    m_flux,                # output: conservative momentum-flux divergence on UGRID
    m, n,                  # transports: m = h*u on UGRID, n = h*v on VGRID
    h,                     # layer thickness on HGRID
    dx_face_u, dy_face_u,  # face lengths around UGRID control volumes
    dArea_u,               # UGRID cell area
    Nx::Int, Ny::Int,
    g::FT,
    hmin::FT,
)
    # Global thread indices
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        # =====================================================
        # Zonal faces (east / west) of the u-cell (i,j)
        # These contribute the normal x-momentum flux:
        #     hu^2 + 1/2 g h^2
        # =====================================================

        # East face: average m from neighboring U points
        m_east = FT(0.5) * (
            stencil(m, i, j,  0, 0, Nx, Ny, UGRID) +
            stencil(m, i, j,  1, 0, Nx, Ny, UGRID)
        )
        # Reconstruct h on the east face from the adjacent H cell
        h_east = max(stencil(h, i, j,  1, 0, Nx, Ny, HGRID), hmin)
        u_east = m_east / h_east
        dy_east = stencil(dy_face_u, i, j,  0, 0, Nx, Ny, GRID0)

        # West face: average m from neighboring U points
        m_west = FT(0.5) * (
            stencil(m, i, j, -1, 0, Nx, Ny, UGRID) +
            stencil(m, i, j,  0, 0, Nx, Ny, UGRID)
        )
        # Reconstruct h on the west face from the adjacent H cell
        h_west = max(stencil(h, i, j,  0, 0, Nx, Ny, HGRID), hmin)
        u_west = m_west / h_west
        dy_west = stencil(dy_face_u, i, j, -1, 0, Nx, Ny, GRID0)

        # Hydrostatic pressure flux contribution on east/west faces
        p_east = FT(0.5) * g * h_east * h_east
        p_west = FT(0.5) * g * h_west * h_west

        # =====================================================
        # Meridional faces (north / south) of the u-cell (i,j)
        # These contribute the transverse flux:
        #     huv
        # =====================================================

        # North face
        n_north = FT(0.5) * (
            stencil(n, i, j,  0, 0, Nx, Ny, VGRID) +
            stencil(n, i, j,  1, 0, Nx, Ny, VGRID)
        )
        m_north = FT(0.5) * (
            stencil(m, i, j,  0, 0, Nx, Ny, UGRID) +
            stencil(m, i, j,  0, 1, Nx, Ny, UGRID)
        )
        # Corner thickness from 4-point average around the north face
        h_north = FT(0.25) * (
            stencil(h, i, j,  0, 0, Nx, Ny, HGRID) +
            stencil(h, i, j,  1, 0, Nx, Ny, HGRID) +
            stencil(h, i, j,  0, 1, Nx, Ny, HGRID) +
            stencil(h, i, j,  1, 1, Nx, Ny, HGRID)
        )
        h_north = max(h_north, hmin)
        v_north = n_north / h_north
        dx_north = stencil(dx_face_u, i, j,  0, 0, Nx, Ny, GRID0)

        # South face
        n_south = FT(0.5) * (
            stencil(n, i, j,  0, -1, Nx, Ny, VGRID) +
            stencil(n, i, j,  1, -1, Nx, Ny, VGRID)
        )
        m_south = FT(0.5) * (
            stencil(m, i, j,  0,  0, Nx, Ny, UGRID) +
            stencil(m, i, j,  0, -1, Nx, Ny, UGRID)
        )
        # Corner thickness from 4-point average around the south face
        h_south = FT(0.25) * (
            stencil(h, i, j,  0,  0, Nx, Ny, HGRID) +
            stencil(h, i, j,  1,  0, Nx, Ny, HGRID) +
            stencil(h, i, j,  0, -1, Nx, Ny, HGRID) +
            stencil(h, i, j,  1, -1, Nx, Ny, HGRID)
        )
        h_south = max(h_south, hmin)
        v_south = n_south / h_south
        dx_south = stencil(dx_face_u, i, j,  0, -1, Nx, Ny, GRID0)

        # Local UGRID control-volume area
        areaU = stencil(dArea_u, i, j, 0, 0, Nx, Ny, GRID0)

        # =====================================================
        # Conservative momentum-flux divergence
        # =====================================================

        # Zonal face fluxes: (hu^2 + 1/2 g h^2) * dy
        # F_west = (u_west * m_west + p_west) * dy_west
        # F_east = (u_east * m_east + p_east) * dy_east
        # x_comp = (F_west - F_east) / areaU

        # # Meridional face fluxes: (huv) * dx
        # F_south = (v_south * m_south) * dx_south
        # F_north = (v_north * m_north) * dx_north
        # y_comp = (F_south - F_north) / areaU

        # Zonal face fluxes: (hu^2 + 1/2 g h^2) * dy
        F_west = (u_west * m_west) * dy_west
        F_east = (u_east * m_east) * dy_east
        x_comp = (F_west - F_east) / areaU

        # Meridional face fluxes: (huv) * dx
        F_south = (v_south * m_south) * dx_south
        F_north = (v_north * m_north) * dx_north
        y_comp = (F_south - F_north) / areaU

        # Total conservative RHS contribution at this u-cell
        m_flux[i, j] = x_comp + y_comp
    end

    return
end


# ---------------------------------------------------------
# Centered conservative momentum-flux divergence
# for n = h*v on v-cells (VGRID)
# ---------------------------------------------------------

"""
    k_calc_flux_n_for_vcell!(n_flux, m, n, h,
                             dx_face_v, dy_face_v, dArea_v,
                             Nx, Ny, g, hmin)

Compute the centered finite-volume conservative flux-divergence term
for meridional momentum `n = h*v` on C-grid v-cells (VGRID), including
metric factors and the hydrostatic pressure flux.

This kernel evaluates the discrete divergence of the y-momentum flux
tensor in conservative form:

    ∂x(huv) + ∂y(hv² + 1/2 g h²)

Using finite-volume notation, the returned quantity is

    n_flux[i,j] = (F_west - F_east + F_south - F_north) / areaV

where

- `F_x = (huv) * dy`             is the zonal face flux
- `F_y = (hv² + 1/2 g h²) * dx`  is the meridional face flux

Notes
-----
- `m` is stored on UGRID and represents `h*u`.
- `n` is stored on VGRID and represents `h*v`.
- `h` is stored on HGRID.
- Face/corner values are obtained by centered interpolation.
- Thickness used in velocity reconstruction is clipped below by `hmin`
  to avoid division by very small values.
"""
function k_calc_flux_n_for_vcell!(
    n_flux,               # output: conservative momentum-flux divergence on VGRID
    m, n,                 # transports: m = h*u on UGRID, n = h*v on VGRID
    h,                    # layer thickness on HGRID
    dx_face_v, dy_face_v, # face lengths around VGRID control volumes
    dArea_v,              # VGRID cell area
    Nx::Int, Ny::Int,
    g::FT,
    hmin::FT,
)
    # Global thread indices
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        # =====================================================
        # Zonal faces (east / west) of the v-cell (i,j)
        # These contribute the transverse flux:
        #     huv
        # =====================================================

        # East face
        n_east = FT(0.5) * (
            stencil(n, i, j,  0, 0, Nx, Ny, VGRID) +
            stencil(n, i, j,  1, 0, Nx, Ny, VGRID)
        )
        m_east = FT(0.5) * (
            stencil(m, i, j,  0, 0, Nx, Ny, UGRID) +
            stencil(m, i, j,  0, 1, Nx, Ny, UGRID)
        )
        # Corner thickness from 4-point average around east face
        h_east = FT(0.25) * (
            stencil(h, i, j,  0, 0, Nx, Ny, HGRID) +
            stencil(h, i, j,  0, 1, Nx, Ny, HGRID) +
            stencil(h, i, j,  1, 0, Nx, Ny, HGRID) +
            stencil(h, i, j,  1, 1, Nx, Ny, HGRID)
        )
        h_east = max(h_east, hmin)
        u_east = m_east / h_east
        dy_east = stencil(dy_face_v, i, j,  0, 0, Nx, Ny, GRID0)

        # West face
        n_west = FT(0.5) * (
            stencil(n, i, j, -1, 0, Nx, Ny, VGRID) +
            stencil(n, i, j,  0, 0, Nx, Ny, VGRID)
        )
        m_west = FT(0.5) * (
            stencil(m, i, j, -1, 0, Nx, Ny, UGRID) +
            stencil(m, i, j, -1, 1, Nx, Ny, UGRID)
        )
        # Corner thickness from 4-point average around west face
        h_west = FT(0.25) * (
            stencil(h, i, j, -1, 0, Nx, Ny, HGRID) +
            stencil(h, i, j, -1, 1, Nx, Ny, HGRID) +
            stencil(h, i, j,  0, 0, Nx, Ny, HGRID) +
            stencil(h, i, j,  0, 1, Nx, Ny, HGRID)
        )
        h_west = max(h_west, hmin)
        u_west = m_west / h_west
        dy_west = stencil(dy_face_v, i, j, -1, 0, Nx, Ny, GRID0)

        # =====================================================
        # Meridional faces (north / south) of the v-cell (i,j)
        # These contribute the normal y-momentum flux:
        #     hv^2 + 1/2 g h^2
        # =====================================================

        # North face
        n_north = FT(0.5) * (
            stencil(n, i, j,  0, 0, Nx, Ny, VGRID) +
            stencil(n, i, j,  0, 1, Nx, Ny, VGRID)
        )
        h_north = max(stencil(h, i, j,  0, 1, Nx, Ny, HGRID), hmin)
        v_north = n_north / h_north
        dx_north = stencil(dx_face_v, i, j,  0, 0, Nx, Ny, GRID0)

        # South face
        n_south = FT(0.5) * (
            stencil(n, i, j,  0, -1, Nx, Ny, VGRID) +
            stencil(n, i, j,  0,  0, Nx, Ny, VGRID)
        )
        h_south = max(stencil(h, i, j,  0, 0, Nx, Ny, HGRID), hmin)
        v_south = n_south / h_south
        dx_south = stencil(dx_face_v, i, j,  0, -1, Nx, Ny, GRID0)

        # Hydrostatic pressure flux contribution on north/south faces
        p_north = FT(0.5) * g * h_north * h_north
        p_south = FT(0.5) * g * h_south * h_south

        # Local VGRID control-volume area
        areaV = stencil(dArea_v, i, j, 0, 0, Nx, Ny, GRID0)

        # =====================================================
        # Conservative momentum-flux divergence
        # =====================================================

        # Zonal face fluxes: (huv) * dy
        # F_west = (u_west * n_west) * dy_west
        # F_east = (u_east * n_east) * dy_east
        # x_comp = (F_west - F_east) / areaV

        # # Meridional face fluxes: (hv^2 + 1/2 g h^2) * dx
        # F_south = (v_south * n_south + p_south) * dx_south
        # F_north = (v_north * n_north + p_north) * dx_north
        # y_comp = (F_south - F_north) / areaV


        # Zonal face fluxes: (huv) * dy
        F_west = (u_west * n_west) * dy_west
        F_east = (u_east * n_east) * dy_east
        x_comp = (F_west - F_east) / areaV

        # Meridional face fluxes: (hv^2 + 1/2 g h^2) * dx
        F_south = (v_south * n_south) * dx_south
        F_north = (v_north * n_north) * dx_north
        y_comp = (F_south - F_north) / areaV

        # Total conservative RHS contribution at this v-cell
        n_flux[i, j] = x_comp + y_comp
    end

    return
end