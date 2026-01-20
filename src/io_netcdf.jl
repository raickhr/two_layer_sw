# io_netcdf.jl
#
# NetCDF output utilities for the 2-layer rotating shallow-water model.
#
# Provides:
#   - NetcdfConfig              : output configuration & counters
#   - init_netcdf               : create file, define dims/vars, write coordinates
#   - append_netcdf             : append one snapshot at time index ti
#   - setup_netcdf_if_needed!   : build NetcdfConfig, initialize file if enabled
#   - write_state!              : write state on schedule, updating NetcdfConfig
#
# Assumptions:
#   - FT        : floating type (Float32/Float64) defined in module scope
#   - Params    : parameter struct with Nx, Ny, H1, H2, etc.
#   - State     : model state (see stateVars.jl)
#   - Grid      : grid metrics with lon_u, lat_u, lon_v, lat_v, lon_h, lat_h
#   - NCDatasets: available (using NCDatasets)

using NCDatasets

# ============================================================
# NetCDF output configuration
# ============================================================

"""
    NetcdfConfig

Lightweight configuration / state for NetCDF output.

Fields
------
- `fname::String`          : NetCDF file name
- `save_interval::Float64` : output interval in seconds
- `t::Float64`             : current simulation time (seconds)
- `save_idx::Int`          : next NetCDF time index (1-based)
- `next_save::Float64`     : next scheduled output time (seconds)
- `do_save::Bool`          : whether NetCDF output is enabled

Notes
-----
- This is an immutable struct. Functions like `write_state!` return a
  **new** NetcdfConfig with updated time counters — callers must reassign:

      netcfg = write_state!(netcfg, current_time, state, p)
"""
struct NetcdfConfig
    fname::String
    save_interval::Float64
    t::Float64
    save_idx::Int
    next_save::Float64
    do_save::Bool
end

# ============================================================
# NetCDF file initialization
# ============================================================

"""
    init_netcdf(fname, p, lon_u, lat_u, lon_v, lat_v, lon_h, lat_h)

Create a new NetCDF file and define:

- Dimensions:
    * lon_u, lat_u
    * lon_v, lat_v
    * lon_h, lat_h
    * time (unlimited)

- Coordinate variables:
    * lon_u, lat_u, lon_v, lat_v, lon_h, lat_h
    * time (seconds)

- Data variables:
    * H, h1, h2, eta, xi   (h-points)
    * Ubt, u1, u2          (u-points)
    * Vbt, v1, v2          (v-points)

Coordinates are written from the supplied arrays (CPU or GPU).
"""
function init_netcdf(
    fname::String,
    p::Params,
    lon_u, lat_u,
    lon_v, lat_v,
    lon_h, lat_h,
)
    ds = NCDataset(fname, "c")

    # Dimensions
    defDim(ds, "lon_u", p.Nx)
    defDim(ds, "lat_u", p.Ny)

    defDim(ds, "lon_v", p.Nx)
    defDim(ds, "lat_v", p.Ny)

    defDim(ds, "lon_h", p.Nx)
    defDim(ds, "lat_h", p.Ny)

    defDim(ds, "time", Inf)

    # Coordinate variables
    lon_u_ds = defVar(ds, "lon_u", FT, ("lon_u",))
    lon_u_ds.attrib["units"]     = "degrees_east"
    lon_u_ds.attrib["long_name"] = "longitude at u points"

    lat_u_ds = defVar(ds, "lat_u", FT, ("lat_u",))
    lat_u_ds.attrib["units"]     = "degrees_north"
    lat_u_ds.attrib["long_name"] = "latitude at u points"

    lon_v_ds = defVar(ds, "lon_v", FT, ("lon_v",))
    lon_v_ds.attrib["units"]     = "degrees_east"
    lon_v_ds.attrib["long_name"] = "longitude at v points"

    lat_v_ds = defVar(ds, "lat_v", FT, ("lat_v",))
    lat_v_ds.attrib["units"]     = "degrees_north"
    lat_v_ds.attrib["long_name"] = "latitude at v points"

    lon_h_ds = defVar(ds, "lon_h", FT, ("lon_h",))
    lon_h_ds.attrib["units"]     = "degrees_east"
    lon_h_ds.attrib["long_name"] = "longitude at h points"

    lat_h_ds = defVar(ds, "lat_h", FT, ("lat_h",))
    lat_h_ds.attrib["units"]     = "degrees_north"
    lat_h_ds.attrib["long_name"] = "latitude at h points"

    vtime = defVar(ds, "time", Float64, ("time",))
    vtime.attrib["units"]     = "seconds"
    vtime.attrib["long_name"] = "simulation time"

    # Write coordinate values (force CPU arrays)
    lon_u_ds[:] = Array(lon_u);  lat_u_ds[:] = Array(lat_u)
    lon_v_ds[:] = Array(lon_v);  lat_v_ds[:] = Array(lat_v)
    lon_h_ds[:] = Array(lon_h);  lat_h_ds[:] = Array(lat_h)

    # -------------------------
    # Fields at h-points
    # -------------------------
    ds_H = defVar(ds, "H", FT, ("lon_h", "lat_h", "time"))
    ds_H.attrib["units"]     = "m"
    ds_H.attrib["long_name"] = "water column thickness"

    ds_h1 = defVar(ds, "h1", FT, ("lon_h", "lat_h", "time"))
    ds_h1.attrib["units"]     = "m"
    ds_h1.attrib["long_name"] = "top-layer thickness"

    ds_h2 = defVar(ds, "h2", FT, ("lon_h", "lat_h", "time"))
    ds_h2.attrib["units"]     = "m"
    ds_h2.attrib["long_name"] = "bottom-layer thickness"

    ds_eta = defVar(ds, "eta", FT, ("lon_h", "lat_h", "time"))
    ds_eta.attrib["units"]     = "m"
    ds_eta.attrib["long_name"] = "sea surface height"

    ds_xi = defVar(ds, "xi", FT, ("lon_h", "lat_h", "time"))
    ds_xi.attrib["units"]     = "m"
    ds_xi.attrib["long_name"] = "interface displacement"

    # -------------------------
    # Fields at u-points
    # -------------------------
    ds_Ubt = defVar(ds, "Ubt", FT, ("lon_u", "lat_u", "time"))
    ds_Ubt.attrib["units"]     = "m s-1"
    ds_Ubt.attrib["long_name"] = "barotropic zonal velocity"

    ds_u1 = defVar(ds, "u1", FT, ("lon_u", "lat_u", "time"))
    ds_u1.attrib["units"]     = "m s-1"
    ds_u1.attrib["long_name"] = "top-layer zonal velocity"

    ds_u2 = defVar(ds, "u2", FT, ("lon_u", "lat_u", "time"))
    ds_u2.attrib["units"]     = "m s-1"
    ds_u2.attrib["long_name"] = "bottom-layer zonal velocity"

    # -------------------------
    # Fields at v-points
    # -------------------------
    ds_Vbt = defVar(ds, "Vbt", FT, ("lon_v", "lat_v", "time"))
    ds_Vbt.attrib["units"]     = "m s-1"
    ds_Vbt.attrib["long_name"] = "barotropic meridional velocity"

    ds_v1 = defVar(ds, "v1", FT, ("lon_v", "lat_v", "time"))
    ds_v1.attrib["units"]     = "m s-1"
    ds_v1.attrib["long_name"] = "top-layer meridional velocity"

    ds_v2 = defVar(ds, "v2", FT, ("lon_v", "lat_v", "time"))
    ds_v2.attrib["units"]     = "m s-1"
    ds_v2.attrib["long_name"] = "bottom-layer meridional velocity"

    close(ds)
    return nothing
end

# ============================================================
# Append one snapshot
# ============================================================

"""
    append_netcdf(fname, ti, tsec; H, h1, h2, eta, xi, U, u1, u2, V, v1, v2)

Append one time snapshot to file `fname` at time index `ti` and time
coordinate `tsec` (seconds).

All field arguments can be GPU arrays; they are converted to CPU with `Array(...)`
before writing.
"""
function append_netcdf(
    fname::String,
    ti::Int,
    tsec::Float64;
    H, h1, h2, eta, xi,
    U, u1, u2,
    V, v1, v2,
)
    ds = NCDataset(fname, "a")
    ds["time"][ti] = tsec

    # Pull from GPU if needed
    AH   = Array(H)
    Ah1  = Array(h1)
    Ah2  = Array(h2)
    Aeta = Array(eta)
    Axi  = Array(xi)

    AU   = Array(U)
    Au1  = Array(u1)
    Au2  = Array(u2)

    AV   = Array(V)
    Av1  = Array(v1)
    Av2  = Array(v2)

    # Write
    ds["H"][:, :, ti]   = AH
    ds["h1"][:, :, ti]  = Ah1
    ds["h2"][:, :, ti]  = Ah2
    ds["eta"][:, :, ti] = Aeta
    ds["xi"][:, :, ti]  = Axi

    ds["u1"][:, :, ti]  = Au1
    ds["u2"][:, :, ti]  = Au2
    ds["Ubt"][:, :, ti] = AU

    ds["v1"][:, :, ti]  = Av1
    ds["v2"][:, :, ti]  = Av2
    ds["Vbt"][:, :, ti] = AV

    close(ds)
    return nothing
end

# ============================================================
# Setup helper
# ============================================================

"""
    setup_netcdf_if_needed!(p, grid, out_netcdf, save_interval) -> NetcdfConfig

Initialize NetCDF output file if `out_netcdf` is provided and `save_interval > 0`.

Always returns a `NetcdfConfig`. If output is disabled, returns a config with
`do_save = false` and an infinite `next_save` time.
"""
function setup_netcdf_if_needed!(
    p::Params,
    grid,
    out_netcdf,
    save_interval,
)::NetcdfConfig

    # Disabled output
    if out_netcdf === nothing || out_netcdf == "" || save_interval <= 0
        @info "NetCDF output disabled"
        return NetcdfConfig("", 0.0, 0.0, 0, Inf, false)
    end

    @info "Initializing NetCDF output file: $out_netcdf"

    # Create file & define variables
    init_netcdf(
        out_netcdf, p,
        grid.lon_u, grid.lat_u,
        grid.lon_v, grid.lat_v,
        grid.lon_h, grid.lat_h,
    )

    # Enabled config
    return NetcdfConfig(
        String(out_netcdf),       # fname
        Float64(save_interval),   # save interval [s]
        0.0,                      # t (simulation time)
        1,                        # save_idx (NetCDF time index starts at 1)
        0.0,                      # next_save time [s]
        true,                     # do_save
    )
end

# ============================================================
# State writing / scheduling
# ============================================================

"""
    write_state!(nc, current_time, state, p) -> NetcdfConfig

Update `nc.t` to `current_time` and, if `current_time ≥ nc.next_save`,
write the full model state to the NetCDF file using `append_netcdf`.

Arguments
---------
- `nc::NetcdfConfig`   : NetCDF output configuration
- `current_time::Real` : simulation time in seconds
- `state::State`       : full model state (prognostic fields)
- `p::Params`          : parameter struct (used for reference layer thicknesses)

Returns
-------
- Updated `NetcdfConfig` with advanced `save_idx` and `next_save` if a write occurs.
  Callers MUST reassign, e.g.:

      netcfg = write_state!(netcfg, current_time, state, p)
"""
function write_state!(
    nc::NetcdfConfig,
    current_time::Real,
    state::State,
    p::Params,
)::NetcdfConfig

    # Refresh simulation time in config (struct is immutable → rebuild)
    nc = NetcdfConfig(
        nc.fname,
        nc.save_interval,
        Float64(current_time),   # t
        nc.save_idx,
        nc.next_save,
        nc.do_save,
    )

    # If disabled → nothing to do
    if !nc.do_save
        return nc
    end

    # Not yet output time
    if nc.t + 1e-12 < nc.next_save
        return nc
    end

    # =======================================================
    # Extract fields from State (GPU) and use Temporary scratch
    # =======================================================
    prog = state.prog
    temp = state.temp

    H  = prog.H
    h1 = prog.h1
    h2 = prog.h2

    M  = prog.M
    N  = prog.N

    m1 = prog.m1
    m2 = prog.m2
    n1 = prog.n1
    n2 = prog.n2

    # Scratch arrays for diagnostics (no new allocations)
    eta = temp.temp_var_x1  # sea surface height
    xi  = temp.temp_var_y1  # interface displacement

    Ubt = temp.temp_var_x2  # barotropic U
    Vbt = temp.temp_var_y2  # barotropic V

    u1  = temp.temp_var_x3  # top-layer u
    v1  = temp.temp_var_y3  # top-layer v

    u2  = temp.temp_var_x4  # bottom-layer u
    v2  = temp.temp_var_y4  # bottom-layer v

    # -------------------------------------------------------
    # Surface height (eta) and interface displacement (xi)
    # -------------------------------------------------------
    η0 = FT(p.H1 + p.H2)  # reference total depth
    η1 = FT(p.H1)         # reference top layer depth

    @. eta = h1 + h2 - η0
    @. xi  = h1 - η1

    # -------------------------------------------------------
    # Velocities (guard against division by very small depths)
    # -------------------------------------------------------
    ϵ = FT(eps(Float64))

    @. Ubt = M / max(H, ϵ)
    @. Vbt = N / max(H, ϵ)

    @. u1 = m1 / max(h1, ϵ)
    @. v1 = n1 / max(h1, ϵ)

    @. u2 = m2 / max(h2, ϵ)
    @. v2 = n2 / max(h2, ϵ)

    # ========================
    # Write to NetCDF snapshot
    # ========================
    append_netcdf(
        nc.fname,
        nc.save_idx,
        nc.t;
        H   = H,
        h1  = h1,
        h2  = h2,
        eta = eta,
        xi  = xi,
        U   = Ubt,
        u1  = u1,
        u2  = u2,
        V   = Vbt,
        v1  = v1,
        v2  = v2,
    )

    # Advance counters (new NetcdfConfig instance)
    return NetcdfConfig(
        nc.fname,
        nc.save_interval,
        nc.t,
        nc.save_idx + 1,
        nc.t + nc.save_interval,
        nc.do_save,
    )
end
