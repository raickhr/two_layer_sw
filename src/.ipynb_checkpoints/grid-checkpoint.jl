# Grid.jl
#
# Grid geometry and metric terms for h-, u-, and v-points.
#
# All geometry is constructed on CPU arrays, then transferred once
# to GPU as CuArrays stored in the Grid struct.

struct Grid
    # Grid variables for h_cell (scalars)
    lon_h::CuArray{FT,1}
    lat_h::CuArray{FT,1}
    dx_n2n_h::CuArray{FT,2}   # distance to east node
    dy_n2n_h::CuArray{FT,2}   # distance to north node
    dx_face_h::CuArray{FT,2}  # north face length
    dy_face_h::CuArray{FT,2}  # east face length
    dArea_h::CuArray{FT,2}    # cell area

    # Grid variables for u_cell (zonal faces)
    lon_u::CuArray{FT,1}
    lat_u::CuArray{FT,1}
    dx_n2n_u::CuArray{FT,2}   # distance to east node
    dy_n2n_u::CuArray{FT,2}   # distance to north node
    dx_face_u::CuArray{FT,2}  # north face length
    dy_face_u::CuArray{FT,2}  # east face length
    dArea_u::CuArray{FT,2}    # cell area
    f_u::CuArray{FT,2}        # coriolis parameter

    # Grid variables for v_cell (meridional faces)
    lon_v::CuArray{FT,1}
    lat_v::CuArray{FT,1}
    dx_n2n_v::CuArray{FT,2}   # distance to east node
    dy_n2n_v::CuArray{FT,2}   # distance to north node
    dx_face_v::CuArray{FT,2}  # north face length
    dy_face_v::CuArray{FT,2}  # east face length
    dArea_v::CuArray{FT,2}    # cell area
    f_v::CuArray{FT,2}        # coriolis parameter

    # Grid variables for z_cell (corner point)
    lon_z::CuArray{FT,1}
    lat_z::CuArray{FT,1}
    dx_n2n_z::CuArray{FT,2}   # distance to east node
    dy_n2n_z::CuArray{FT,2}   # distance to north node
    dx_face_z::CuArray{FT,2}  # north face length
    dy_face_z::CuArray{FT,2}  # east face length
    dArea_z::CuArray{FT,2}    # cell area
end

function build_gridVars(p::Params)::Grid
    Nx, Ny   = p.Nx, p.Ny
    deg2rad  = FT(π) / FT(180.0)
    earthR   = FT(p.earthRadius)
    dlon_rad = FT(p.dlon) * deg2rad
    dlat_rad = FT(p.dlat) * deg2rad

    # -----------------------------
    # Allocate CPU arrays
    # -----------------------------

    # h-cell (scalar points)
    lon_h    = Vector{FT}(undef, Nx)
    lat_h    = Vector{FT}(undef, Ny)
    dx_n2n_h = Array{FT}(undef, Nx, Ny)
    dy_n2n_h = Array{FT}(undef, Nx, Ny)
    dx_face_h = Array{FT}(undef, Nx, Ny)
    dy_face_h = Array{FT}(undef, Nx, Ny)
    dArea_h   = Array{FT}(undef, Nx, Ny)

    # u-cell (zonal faces)
    lon_u    = Vector{FT}(undef, Nx)
    lat_u    = Vector{FT}(undef, Ny)
    dx_n2n_u = Array{FT}(undef, Nx, Ny)
    dy_n2n_u = Array{FT}(undef, Nx, Ny)
    dx_face_u = Array{FT}(undef, Nx, Ny)
    dy_face_u = Array{FT}(undef, Nx, Ny)
    dArea_u   = Array{FT}(undef, Nx, Ny)
    f_u   = Array{FT}(undef, Nx, Ny)

    # v-cell (meridional faces)
    lon_v    = Vector{FT}(undef, Nx)
    lat_v    = Vector{FT}(undef, Ny)
    dx_n2n_v = Array{FT}(undef, Nx, Ny)
    dy_n2n_v = Array{FT}(undef, Nx, Ny)
    dx_face_v = Array{FT}(undef, Nx, Ny)
    dy_face_v = Array{FT}(undef, Nx, Ny)
    dArea_v   = Array{FT}(undef, Nx, Ny)
    f_v   = Array{FT}(undef, Nx, Ny)


    # v-cell (meridional faces)
    lon_z    = Vector{FT}(undef, Nx)
    lat_z    = Vector{FT}(undef, Ny)
    dx_n2n_z = Array{FT}(undef, Nx, Ny)
    dy_n2n_z = Array{FT}(undef, Nx, Ny)
    dx_face_z = Array{FT}(undef, Nx, Ny)
    dy_face_z = Array{FT}(undef, Nx, Ny)
    dArea_z   = Array{FT}(undef, Nx, Ny)

    # -----------------------------
    # Longitudes (1D)
    # -----------------------------
    @inbounds for i in 1:Nx
        lon_base = FT(p.lon1) + FT(p.dlon) * FT(i - 1)
        lon_base_plus_half = lon_base + FT(p.dlon) / FT(2.0)
        lon_h[i] = lon_base
        lon_v[i] = lon_base
        lon_u[i] = lon_base_plus_half
        lon_z[i] = lon_base_plus_half
    end

    

    # -----------------------------
    # Latitudes & meridional metrics
    # dy_n2n and dy_face are constant in i, j-dep only through lat.
    # -----------------------------

    # These are constant everywhere for lat-lon grid
    dy_val = earthR * dlat_rad

    dy_n2n_h .= dy_val
    dy_face_h .= dy_val
    dy_n2n_u .= dy_val
    dy_face_u .= dy_val
    dy_n2n_v .= dy_val
    dy_face_v .= dy_val
    dy_n2n_z .= dy_val
    dy_face_z .= dy_val

    @inbounds for j in 1:Ny
        lat_base = FT(p.lat1) + FT(p.dlat) * FT(j - 1)
        lat_base_plus_half = lat_base + FT(p.dlat) / FT(2.0)
        lat_base_plus_one = lat_base + FT(p.dlat) 

        dx_base = earthR * cos(lat_base * deg2rad) * dlon_rad
        dx_base_plus_half = earthR * cos(lat_base_plus_half * deg2rad) * dlon_rad
        dx_base_plus_one = earthR * cos(lat_base_plus_one * deg2rad) * dlon_rad

        lat_h[j] = lat_base
        lat_u[j] = lat_base
        lat_v[j] = lat_base_plus_half
        lat_z[j] = lat_base_plus_half

        dx_n2n_h[:, j] .= dx_base
        dx_n2n_u[:, j] .= dx_base

        dx_n2n_v[:, j] .= dx_base_plus_half
        dx_n2n_z[:, j] .= dx_base_plus_half

        dx_face_h[:, j] .= dx_base_plus_half
        dx_face_u[:, j] .= dx_base_plus_half

        dx_face_v[:, j] .= dx_base_plus_one        
        dx_face_z[:, j] .= dx_base_plus_one
    end 

    # -----------------------------
    # Cell areas (approximate)
    # -----------------------------
    dArea_h .= dx_n2n_h .* dy_face_h
    dArea_u .= dx_n2n_u .* dy_face_u
    dArea_v .= dx_n2n_v .* dy_face_v
    dArea_z .= dx_n2n_z .* dy_face_z


    # -----------------------------
    # Coriolis Paramter
    # -----------------------------

    Ω = FT(2)*FT(π)/FT(3600 * 24)
    @. f_u = FT(2 * Ω) * sin(lat_u * (FT(pi) / FT(180)))
    @. f_v = FT(2 * Ω) * sin(lat_v * (FT(pi) / FT(180)))

    # -----------------------------
    # Move to GPU (one allocation per field)
    # -----------------------------
    return Grid(
        CuArray(lon_h),   CuArray(lat_h),
        CuArray(dx_n2n_h), CuArray(dy_n2n_h),
        CuArray(dx_face_h), CuArray(dy_face_h),
        CuArray(dArea_h),

        CuArray(lon_u),   CuArray(lat_u),
        CuArray(dx_n2n_u), CuArray(dy_n2n_u),
        CuArray(dx_face_u), CuArray(dy_face_u),
        CuArray(dArea_u), CuArray(f_u),

        CuArray(lon_v),   CuArray(lat_v),
        CuArray(dx_n2n_v), CuArray(dy_n2n_v),
        CuArray(dx_face_v), CuArray(dy_face_v),
        CuArray(dArea_v), CuArray(f_v),

        CuArray(lon_z),   CuArray(lat_z),
        CuArray(dx_n2n_z), CuArray(dy_n2n_z),
        CuArray(dx_face_z), CuArray(dy_face_z),
        CuArray(dArea_z)
    )
end
