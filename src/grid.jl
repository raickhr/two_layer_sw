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

    # Grid variables for v_cell (meridional faces)
    lon_v::CuArray{FT,1}
    lat_v::CuArray{FT,1}
    dx_n2n_v::CuArray{FT,2}   # distance to east node
    dy_n2n_v::CuArray{FT,2}   # distance to north node
    dx_face_v::CuArray{FT,2}  # north face length
    dy_face_v::CuArray{FT,2}  # east face length
    dArea_v::CuArray{FT,2}    # cell area
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

    # v-cell (meridional faces)
    lon_v    = Vector{FT}(undef, Nx)
    lat_v    = Vector{FT}(undef, Ny)
    dx_n2n_v = Array{FT}(undef, Nx, Ny)
    dy_n2n_v = Array{FT}(undef, Nx, Ny)
    dx_face_v = Array{FT}(undef, Nx, Ny)
    dy_face_v = Array{FT}(undef, Nx, Ny)
    dArea_v   = Array{FT}(undef, Nx, Ny)

    # -----------------------------
    # Longitudes (1D)
    # -----------------------------
    @inbounds for i in 1:Nx
        lon_base = FT(p.lon1) + FT(p.dlon) * FT(i - 1)
        lon_h[i] = lon_base
        lon_v[i] = lon_base
        lon_u[i] = lon_base + FT(p.dlon) / FT(2.0)
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

    @inbounds for j in 1:Ny
        lat_h_val = FT(p.lat1) + FT(p.dlat) * FT(j - 1)
        lat_u_val = lat_h_val
        lat_v_val = FT(p.lat1) + FT(p.dlat) * FT(j - 1) + FT(p.dlat) / FT(2.0)

        lat_h[j] = lat_h_val
        lat_u[j] = lat_u_val
        lat_v[j] = lat_v_val

        cos_h = cos(lat_h_val * deg2rad)
        cos_u = cos(lat_u_val * deg2rad)
        cos_v = cos(lat_v_val * deg2rad)

        # h-cell: center metric and faces
        dx_n2n_h[:, j] .= earthR * cos_h * dlon_rad
        dx_face_h[:, j] .= earthR * cos_v * dlon_rad

        # u-cell: zonal faces (u on h-row, v-lat for faces)
        dx_n2n_u[:, j] .= earthR * cos_u * dlon_rad
        dx_face_u[:, j] .= earthR * cos_v * dlon_rad

        # v-cell: meridional faces
        dx_n2n_v[:, j] .= earthR * cos_v * dlon_rad
        # Slight offset for face length if desired (as you had):
        dx_face_v[:, j] .= earthR * cos((lat_v_val + FT(p.dlat)/FT(2.0)) * deg2rad) * dlon_rad
    end

    # -----------------------------
    # Cell areas (approximate)
    # -----------------------------
    dArea_h .= dx_n2n_h .* dy_face_h
    dArea_u .= dx_n2n_u .* dy_face_u
    dArea_v .= dx_n2n_v .* dy_face_v

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
        CuArray(dArea_u),

        CuArray(lon_v),   CuArray(lat_v),
        CuArray(dx_n2n_v), CuArray(dy_n2n_v),
        CuArray(dx_face_v), CuArray(dy_face_v),
        CuArray(dArea_v)
    )
end
