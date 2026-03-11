############################################################
# coriolis_curvature.jl
#
# Semi-implicit Coriolis rotation and curvature (metric)
# terms for the barotropic (external) mode in spherical
# or curvilinear coordinates on a C-grid.
#
# Assumed sign convention for momentum equations:
#
#   u_t = f v - g η_x + ...
#   v_t = -f u - g η_y + ...
#
# so that,
#   - For zonal momentum (hu), curvature term ≈ -h u v tan(φ)/R
#   - For meridional momentum (hv), curvature term ≈ +h u² tan(φ)/R
#
############################################################

function k_add_coriolisforce!(
    m_new, n_new,    # output: hu, hv after Coriolis
    m_star, n_star,  # Intermediate hu, hv from all non-Coriolis terms
    m_old, n_old,    # hu, hv at previous timestep (n)
    lat_u, lat_v,    # latitudes of u and v points (degrees)
    Nx::Int, Ny::Int,
    dt::FT, Ω::FT,
)
    """
    Apply a semi-implicit Coriolis rotation to barotropic momenta.

    At each (i,j), we compute the local Coriolis parameters

        f_u = f(φ_u),  f_v = f(φ_v),
        α_u = dt * f_u, α_v = dt * f_v,
        β_u = α_u^2 / 4, β_v = α_v^2 / 4,

    and update:

        m_new = ( m_star - β_u m_old + α_u n_old_in_u ) / (1 + β_u)
        n_new = ( n_star - β_v n_old - α_v m_old_in_v ) / (1 + β_v)

    where m_old_in_v and n_old_in_u are reconstructions of hu at v-points
    and hv at u-points via simple 4-point averaging around the C-grid cell:

        m_old_in_v(i,j) ≈ average of m_old around v(i,j)
        n_old_in_u(i,j) ≈ average of n_old around u(i,j)

    This approximates a Crank–Nicolson-like treatment of Coriolis in
    flux form while using collocated reconstructions to avoid
    pressure–Coriolis decoupling on the C-grid.

    NOTE:
        All reads of m_old and n_old are done via `stencil(...)`
        so that the same reconstruction machinery is used everywhere.
    """

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        # Local Coriolis parameters at u/v-latitudes
        f_u = coriolis_at_lat(Ω, lat_u[j])  # Coriolis at u-latitude
        f_v = coriolis_at_lat(Ω, lat_v[j])  # Coriolis at v-latitude

        α_u = dt * f_u
        α_v = dt * f_v

        β_u = FT(0.25) * α_u^2
        β_v = FT(0.25) * α_v^2

        #----------------------------------------------------
        # Reconstruct m (hu) at v-points (collocated with n)
        #----------------------------------------------------
        m_old_in_v = FT(0.25) * (
            stencil(m_old, i, j,   0,  0, Nx, Ny, UGRID) +
            stencil(m_old, i, j,   0, +1, Nx, Ny, UGRID) +
            stencil(m_old, i, j,  -1,  0, Nx, Ny, UGRID) +
            stencil(m_old, i, j,  -1, +1, Nx, Ny, UGRID)
        )

        #----------------------------------------------------
        # Reconstruct n (hv) at u-points (collocated with m)
        #----------------------------------------------------
        n_old_in_u = FT(0.25) * (
            stencil(n_old, i, j,   0,  0, Nx, Ny, VGRID) +
            stencil(n_old, i, j,  +1,  0, Nx, Ny, VGRID) +
            stencil(n_old, i, j,   0, -1, Nx, Ny, VGRID) +
            stencil(n_old, i, j,  +1, -1, Nx, Ny, VGRID)
        )

        # Native-point values at the current u/v locations (also via stencil)
        m_old_here = stencil(m_old, i, j,  0, 0, Nx, Ny, UGRID)
        n_old_here = stencil(n_old, i, j,  0, 0, Nx, Ny, VGRID)

        m_star_here = stencil(m_star, i, j,  0, 0, Nx, Ny, UGRID)
        n_star_here = stencil(n_star, i, j,  0, 0, Nx, Ny, VGRID)

        # Semi-implicit rotation update
        m_new[i, j] = (m_star_here - β_u * m_old_here + α_u * n_old_in_u) / (FT(1) + β_u)
        n_new[i, j] = (n_star_here - β_v * n_old_here - α_v * m_old_in_v) / (FT(1) + β_v)
    end

    return
end


function k_calc_curvature_terms!(
    curvature_x, curvature_y,  # output: curvature terms in hu- and hv-equations
    m_old, n_old,              # hu and hv at u and v points (previous timestep)
    h_old_in_u, h_old_in_v,    # h reconstructed at u and v points
    lat_u, lat_v,              # latitudes of u and v points (degrees)
    Nx::Int, Ny::Int,
    earthRadius::FT,
)
    """
    Compute curvature (metric) terms for spherical shallow water
    on a C-grid in flux form.

    Using local velocities

        u = m / h,  v = n / h,

    this kernel approximates:

        curvature_x ≈ h u v tan(φ_u) / R      (zonal momentum, hu-equation)
        curvature_y ≈ -h u^2 tan(φ_v) / R      (meridional momentum, hv-equation)

    where:
        φ_u, φ_v are latitudes of u- and v-points,
        R = earthRadius.

    These are the usual spherical metric terms that arise from the
    covariant derivative in spherical coordinates.
    They should be added to the right-hand side of the momentum
    equations (NOT inside the flux divergence).

    NOTE:
        - h_old_in_u and h_old_in_v are assumed to have been
          reconstructed with a minimum thickness (hmin) to avoid
          division by zero.
        - All reads of m_old, n_old, h_old_in_u, h_old_in_v
          are done via `stencil(...)` using the appropriate
          grid character (UGRID or VGRID).
    """

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    deg2rad = FT(π) / FT(180)

    if i <= Nx && j <= Ny
        #----------------------------------------------------
        # Reconstruction stencil (C-grid cell), conceptually:
        #
        #   |---------------- v(i,j) ---------------|
        #   |                                       |
        #   |                                       |
        #   |                                       |
        # u(i-1,j)            h(·)                u(i,j)
        #   |                                       |
        #   |                                       |
        #   |                                       |
        #   |--------------- v(i,j-1) --------------|
        #
        # Here, h_old_in_u and h_old_in_v already live at u- and v-
        # points; we just sample them with stencil(..., 0,0,...).
        #----------------------------------------------------

        #----------------------------------------------------
        # 1) Local thickness at u/v points (via stencil)
        #----------------------------------------------------
        h_u_here = stencil(h_old_in_u, i, j,  0, 0, Nx, Ny, UGRID)
        h_v_here = stencil(h_old_in_v, i, j,  0, 0, Nx, Ny, VGRID)

        #----------------------------------------------------
        # 2) u, v at native u/v locations
        #----------------------------------------------------
        m_here_u = stencil(m_old, i, j,  0, 0, Nx, Ny, UGRID)
        n_here_v = stencil(n_old, i, j,  0, 0, Nx, Ny, VGRID)

        u_old = m_here_u / h_u_here
        v_old = n_here_v / h_v_here

        #----------------------------------------------------
        # 3) u at v-points: reconstruct m at v, then divide by h_v_here
        #----------------------------------------------------
        m_old_in_v = FT(0.25) * (
            stencil(m_old, i, j,   0,  0, Nx, Ny, UGRID) +
            stencil(m_old, i, j,   0, +1, Nx, Ny, UGRID) +
            stencil(m_old, i, j,  -1,  0, Nx, Ny, UGRID) +
            stencil(m_old, i, j,  -1, +1, Nx, Ny, UGRID)
        )

        u_old_in_v = m_old_in_v / h_v_here

        #----------------------------------------------------
        # 4) v at u-points: reconstruct n at u, then divide by h_u_here
        #----------------------------------------------------
        n_old_in_u = FT(0.25) * (
            stencil(n_old, i, j,   0,  0, Nx, Ny, VGRID) +
            stencil(n_old, i, j,  +1,  0, Nx, Ny, VGRID) +
            stencil(n_old, i, j,   0, -1, Nx, Ny, VGRID) +
            stencil(n_old, i, j,  +1, -1, Nx, Ny, VGRID)
        )

        v_old_in_u = n_old_in_u / h_u_here

        #----------------------------------------------------
        # 5) Curvature terms
        #----------------------------------------------------
        # Zonal (hu) equation:
        #   (hu)_t +=  h u v tan(φ_u) / R
        curvature_x[i, j] = h_u_here * u_old * v_old_in_u *
                            tan(lat_u[j] * deg2rad) / earthRadius

        # Meridional (hv) equation:
        #   (hv)_t += - h u^2 tan(φ_v) / R
        curvature_y[i, j] = -h_v_here * u_old_in_v * u_old_in_v *
                            tan(lat_v[j] * deg2rad) / earthRadius
    end

    return
end
