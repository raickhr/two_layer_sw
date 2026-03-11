# io_netcdf.jl
#
# NetCDF output utilities for the 2-layer rotating shallow-water model.
#
# Prognostic output:
#   - NetcdfConfig
#   - init_netcdf
#   - append_netcdf
#   - setup_netcdf_if_needed!
#   - write_state!
#
# Diagnostic output (SEPARATE NetCDF FILE):
#   - init_netcdf_diag
#   - append_netcdf_diag
#
# Grid → dim mapping:
#   UGRID (u) : (lon_u, lat_u)
#   VGRID (v) : (lon_v, lat_v)
#   HGRID (h) : (lon_h, lat_h)
#   ZGRID (z) : (lon_u, lat_v)

using NCDatasets

# ============================================================
# NetCDF output configuration (prognostic file)
# ============================================================

struct NetcdfConfig
    fname::String       # prognostic file
    diag_fname::String  # diagnostic file ("" if disabled)
    save_interval::Float64
    t::Float64
    save_idx::Int
    next_save::Float64
    do_save::Bool
end

# ============================================================
# Prognostic NetCDF file initialization
# ============================================================

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
    lat_u_ds = defVar(ds, "lat_u", FT, ("lat_u",))
    lon_v_ds = defVar(ds, "lon_v", FT, ("lon_v",))
    lat_v_ds = defVar(ds, "lat_v", FT, ("lat_v",))
    lon_h_ds = defVar(ds, "lon_h", FT, ("lon_h",))
    lat_h_ds = defVar(ds, "lat_h", FT, ("lat_h",))

    lon_u_ds.attrib["units"]     = "degrees_east"
    lon_u_ds.attrib["long_name"] = "longitude at u points"
    lat_u_ds.attrib["units"]     = "degrees_north"
    lat_u_ds.attrib["long_name"] = "latitude at u points"

    lon_v_ds.attrib["units"]     = "degrees_east"
    lon_v_ds.attrib["long_name"] = "longitude at v points"
    lat_v_ds.attrib["units"]     = "degrees_north"
    lat_v_ds.attrib["long_name"] = "latitude at v points"

    lon_h_ds.attrib["units"]     = "degrees_east"
    lon_h_ds.attrib["long_name"] = "longitude at h points"
    lat_h_ds.attrib["units"]     = "degrees_north"
    lat_h_ds.attrib["long_name"] = "latitude at h points"

    vtime = defVar(ds, "time", Float64, ("time",))
    vtime.attrib["units"]     = "seconds"
    vtime.attrib["long_name"] = "simulation time"

    # Coordinate values
    lon_u_ds[:] = Array(lon_u)
    lat_u_ds[:] = Array(lat_u)
    lon_v_ds[:] = Array(lon_v)
    lat_v_ds[:] = Array(lat_v)
    lon_h_ds[:] = Array(lon_h)
    lat_h_ds[:] = Array(lat_h)

    # h-points
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

    # u-points
    ds_Ubt = defVar(ds, "Ubt", FT, ("lon_u", "lat_u", "time"))
    ds_Ubt.attrib["units"]     = "m s-1"
    ds_Ubt.attrib["long_name"] = "barotropic zonal velocity"

    ds_u1 = defVar(ds, "u1", FT, ("lon_u", "lat_u", "time"))
    ds_u1.attrib["units"]     = "m s-1"
    ds_u1.attrib["long_name"] = "top-layer zonal velocity"

    ds_u2 = defVar(ds, "u2", FT, ("lon_u", "lat_u", "time"))
    ds_u2.attrib["units"]     = "m s-1"
    ds_u2.attrib["long_name"] = "bottom-layer zonal velocity"

    # v-points
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
# Prognostic append
# ============================================================

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
# Diagnostics NetCDF (SEPARATE FILE)
# ============================================================

function init_netcdf_diag(
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
    lat_u_ds = defVar(ds, "lat_u", FT, ("lat_u",))
    lon_v_ds = defVar(ds, "lon_v", FT, ("lon_v",))
    lat_v_ds = defVar(ds, "lat_v", FT, ("lat_v",))
    lon_h_ds = defVar(ds, "lon_h", FT, ("lon_h",))
    lat_h_ds = defVar(ds, "lat_h", FT, ("lat_h",))

    lon_u_ds.attrib["units"]     = "degrees_east"
    lon_u_ds.attrib["long_name"] = "longitude at u points"
    lat_u_ds.attrib["units"]     = "degrees_north"
    lat_u_ds.attrib["long_name"] = "latitude at u points"

    lon_v_ds.attrib["units"]     = "degrees_east"
    lon_v_ds.attrib["long_name"] = "longitude at v points"
    lat_v_ds.attrib["units"]     = "degrees_north"
    lat_v_ds.attrib["long_name"] = "latitude at v points"

    lon_h_ds.attrib["units"]     = "degrees_east"
    lon_h_ds.attrib["long_name"] = "longitude at h points"
    lat_h_ds.attrib["units"]     = "degrees_north"
    lat_h_ds.attrib["long_name"] = "latitude at h points"

    vtime = defVar(ds, "time", Float64, ("time",))
    vtime.attrib["units"]     = "seconds"
    vtime.attrib["long_name"] = "simulation time"

    lon_u_ds[:] = Array(lon_u)
    lat_u_ds[:] = Array(lat_u)
    lon_v_ds[:] = Array(lon_v)
    lat_v_ds[:] = Array(lat_v)
    lon_h_ds[:] = Array(lon_h)
    lat_h_ds[:] = Array(lat_h)

    # H-grid diagnostics (lon_h, lat_h)
    for (nm, longnm, units) in [
        ("gradx_u1",  "d(u1)/dx (H-grid view)",           "s-1"),
        ("gradx_u2",  "d(u2)/dx (H-grid view)",           "s-1"),
        ("grady_v1",  "d(v1)/dy (H-grid view)",           "s-1"),
        ("grady_v2",  "d(v2)/dy (H-grid view)",           "s-1"),
        ("pressure1", "pressure (H-grid view), layer 1",  "m2 s-2"),
        ("pressure2", "pressure (H-grid view), layer 2",  "m2 s-2"),
    ]
        v = defVar(ds, nm, FT, ("lon_h", "lat_h", "time"))
        v.attrib["long_name"] = longnm
        v.attrib["units"]     = units
    end

    # Z-grid diagnostics (lon_u, lat_v)
    for (nm, longnm, units) in [
        ("grady_u1", "d(u1)/dy (Z-grid view)",                    "s-1"),
        ("grady_u2", "d(u2)/dy (Z-grid view)",                    "s-1"),
        ("gradx_v1", "d(v1)/dx (Z-grid view)",                    "s-1"),
        ("gradx_v2", "d(v2)/dx (Z-grid view)",                    "s-1"),
        ("zeta1",    "relative vorticity, layer 1 (Z-grid view)", "s-1"),
        ("zeta2",    "relative vorticity, layer 2 (Z-grid view)", "s-1"),
    ]
        v = defVar(ds, nm, FT, ("lon_u", "lat_v", "time"))
        v.attrib["long_name"] = longnm
        v.attrib["units"]     = units
    end

    # Pressure gradients: UGRID and VGRID
    for (nm, longnm, units, dims) in [
        ("gradx_p1", "d(p1)/dx (U-grid view)", "m s-2", ("lon_u", "lat_u", "time")),
        ("gradx_p2", "d(p2)/dx (U-grid view)", "m s-2", ("lon_u", "lat_u", "time")),
        ("grady_p1", "d(p1)/dy (V-grid view)", "m s-2", ("lon_v", "lat_v", "time")),
        ("grady_p2", "d(p2)/dy (V-grid view)", "m s-2", ("lon_v", "lat_v", "time")),
    ]
        v = defVar(ds, nm, FT, dims)
        v.attrib["long_name"] = longnm
        v.attrib["units"]     = units
    end

    # U-grid diagnostics (lon_u, lat_u)
    for (nm, longnm, units) in [
        ("advec_u1",     "advection tendency of u1",        "m s-2"),
        ("advec_u2",     "advection tendency of u2",        "m s-2"),
        ("nu_u1",        "viscosity coefficient nu for u1", "m2 s-1"),
        ("nu_u2",        "viscosity coefficient nu for u2", "m2 s-1"),
        ("diff_u1",      "diffusive tendency of u1",        "m s-2"),
        ("diff_u2",      "diffusive tendency of u2",        "m s-2"),
        ("curv_adv_u1",  "metric advection term for u1",    "m s-2"),
        ("curv_adv_u2",  "metric advection term for u2",    "m s-2"),
        ("curv_visc_u1", "metric viscosity term for u1",    "m s-2"),
        ("curv_visc_u2", "metric viscosity term for u2",    "m s-2"),
        ("cor_u1",       "Coriolis term for u1",            "m s-2"),
        ("cor_u2",       "Coriolis term for u2",            "m s-2"),
    ]
        v = defVar(ds, nm, FT, ("lon_u", "lat_u", "time"))
        v.attrib["long_name"] = longnm
        v.attrib["units"]     = units
    end

    # V-grid diagnostics (lon_v, lat_v)
    for (nm, longnm, units) in [
        ("advec_v1",     "advection tendency of v1",        "m s-2"),
        ("advec_v2",     "advection tendency of v2",        "m s-2"),
        ("nu_v1",        "viscosity coefficient nu for v1", "m2 s-1"),
        ("nu_v2",        "viscosity coefficient nu for v2", "m2 s-1"),
        ("diff_v1",      "diffusive tendency of v1",        "m s-2"),
        ("diff_v2",      "diffusive tendency of v2",        "m s-2"),
        ("curv_adv_v1",  "metric advection term for v1",    "m s-2"),
        ("curv_adv_v2",  "metric advection term for v2",    "m s-2"),
        ("curv_visc_v1", "metric viscosity term for v1",    "m s-2"),
        ("curv_visc_v2", "metric viscosity term for v2",    "m s-2"),
        ("cor_v1",       "Coriolis term for v1",            "m s-2"),
        ("cor_v2",       "Coriolis term for v2",            "m s-2"),
    ]
        v = defVar(ds, nm, FT, ("lon_v", "lat_v", "time"))
        v.attrib["long_name"] = longnm
        v.attrib["units"]     = units
    end

    close(ds)
    return nothing
end

function append_netcdf_diag(
    fname::String,
    ti::Int,
    tsec::Float64,
    diag1::Diagnostic,
    diag2::Diagnostic,
)
    ds = NCDataset(fname, "a")
    ds["time"][ti] = tsec

    # H-grid: (lon_h, lat_h)
    ds["gradx_u1"][:, :, ti]  = Array(diag1.gradx_u)
    ds["gradx_u2"][:, :, ti]  = Array(diag2.gradx_u)
    ds["grady_v1"][:, :, ti]  = Array(diag1.grady_v)
    ds["grady_v2"][:, :, ti]  = Array(diag2.grady_v)
    ds["pressure1"][:, :, ti] = Array(diag1.pressure)
    ds["pressure2"][:, :, ti] = Array(diag2.pressure)

    # Z-grid: (lon_u, lat_v)
    ds["grady_u1"][:, :, ti]  = Array(diag1.grady_u)
    ds["grady_u2"][:, :, ti]  = Array(diag2.grady_u)
    ds["gradx_v1"][:, :, ti]  = Array(diag1.gradx_v)
    ds["gradx_v2"][:, :, ti]  = Array(diag2.gradx_v)
    ds["zeta1"][:, :, ti]     = Array(diag1.zeta)
    ds["zeta2"][:, :, ti]     = Array(diag2.zeta)

    # Pressure gradients
    ds["gradx_p1"][:, :, ti]  = Array(diag1.gradx_pres)
    ds["gradx_p2"][:, :, ti]  = Array(diag2.gradx_pres)
    ds["grady_p1"][:, :, ti]  = Array(diag1.grady_pres)
    ds["grady_p2"][:, :, ti]  = Array(diag2.grady_pres)

    # U-grid
    ds["advec_u1"][:, :, ti]      = Array(diag1.advec_term_u)
    ds["advec_u2"][:, :, ti]      = Array(diag2.advec_term_u)
    ds["nu_u1"][:, :, ti]         = Array(diag1.viscosity_u)
    ds["nu_u2"][:, :, ti]         = Array(diag2.viscosity_u)
    ds["diff_u1"][:, :, ti]       = Array(diag1.viscous_term_u)
    ds["diff_u2"][:, :, ti]       = Array(diag2.viscous_term_u)
    ds["curv_adv_u1"][:, :, ti]   = Array(diag1.advx_curv)
    ds["curv_adv_u2"][:, :, ti]   = Array(diag2.advx_curv)
    ds["curv_visc_u1"][:, :, ti]  = Array(diag1.visx_curv)
    ds["curv_visc_u2"][:, :, ti]  = Array(diag2.visx_curv)
    ds["cor_u1"][:, :, ti]        = Array(diag1.coriolis_term_u)
    ds["cor_u2"][:, :, ti]        = Array(diag2.coriolis_term_u)

    # V-grid
    ds["advec_v1"][:, :, ti]      = Array(diag1.advec_term_v)
    ds["advec_v2"][:, :, ti]      = Array(diag2.advec_term_v)
    ds["nu_v1"][:, :, ti]         = Array(diag1.viscosity_v)
    ds["nu_v2"][:, :, ti]         = Array(diag2.viscosity_v)
    ds["diff_v1"][:, :, ti]       = Array(diag1.viscous_term_v)
    ds["diff_v2"][:, :, ti]       = Array(diag2.viscous_term_v)
    ds["curv_adv_v1"][:, :, ti]   = Array(diag1.advy_curv)
    ds["curv_adv_v2"][:, :, ti]   = Array(diag2.advy_curv)
    ds["curv_visc_v1"][:, :, ti]  = Array(diag1.visy_curv)
    ds["curv_visc_v2"][:, :, ti]  = Array(diag2.visy_curv)
    ds["cor_v1"][:, :, ti]        = Array(diag1.coriolis_term_v)
    ds["cor_v2"][:, :, ti]        = Array(diag2.coriolis_term_v)

    close(ds)
    return nothing
end

# ============================================================
# Setup helper (prognostic + optional diagnostic files)
# ============================================================

function setup_netcdf_if_needed!(
    p::Params,
    grid,
    out_netcdf,
    diag_out_netcdf,
    save_interval,
)::NetcdfConfig

    if out_netcdf === nothing || out_netcdf == "" || save_interval <= 0
        @info "NetCDF output (prognostic) disabled"
        return NetcdfConfig("", "", 0.0, 0.0, 0, Inf, false)
    end

    @info "Initializing prognostic NetCDF file: $out_netcdf"

    init_netcdf(
        out_netcdf, p,
        grid.lon_u, grid.lat_u,
        grid.lon_v, grid.lat_v,
        grid.lon_h, grid.lat_h,
    )

    if diag_out_netcdf !== nothing && diag_out_netcdf != ""
        @info "Initializing diagnostic NetCDF file: $diag_out_netcdf"
        init_netcdf_diag(
            diag_out_netcdf, p,
            grid.lon_u, grid.lat_u,
            grid.lon_v, grid.lat_v,
            grid.lon_h, grid.lat_h,
        )
    else
        @info "Diagnostic NetCDF output disabled"
        diag_out_netcdf = ""
    end

    return NetcdfConfig(
        String(out_netcdf),
        String(diag_out_netcdf),
        Float64(save_interval),
        0.0,
        1,
        0.0,
        true,
    )
end

# ============================================================
# State writing / scheduling (prognostic, plus optional diag)
# ============================================================

function write_state!(
    nc::NetcdfConfig,
    current_time::Real,
    state::State,
    p::Params,
    threads2::NTuple{2,Int},
    blocks2::NTuple{2,Int},
)::NetcdfConfig

    # Update time in config
    nc = NetcdfConfig(
        nc.fname,
        nc.diag_fname,
        nc.save_interval,
        Float64(current_time),
        nc.save_idx,
        nc.next_save,
        nc.do_save,
    )

    if !nc.do_save
        return nc
    end

    if nc.t + 1e-12 < nc.next_save
        return nc
    end

    ϵ       = FT(eps(Float64))
    hmin    = FT(p.hmin)
    hmin_bt = max(hmin, ϵ)

    prog1 = state.prog1
    prog2 = state.prog2

    diag1 = state.diag1
    diag2 = state.diag2

    temp = state.temp

    h1 = prog1.h
    h2 = prog2.h

    m1 = prog1.m
    m2 = prog2.m

    n1 = prog1.n
    n2 = prog2.n

    u1 = diag1.u
    v1 = diag1.v

    u2 = diag2.u
    v2 = diag2.v

    h_in_u = state.intp.h_in_u
    h_in_v = state.intp.h_in_v

    # Layer 1 velocities (UGRID / VGRID)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h_in_u, h1, p.Nx, p.Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h_in_v, h1, p.Nx, p.Ny)

    @. h_in_u = max(h_in_u, hmin)
    @. h_in_v = max(h_in_v, hmin)

    @. u1 = m1 / h_in_u
    @. v1 = n1 / h_in_v

    # Layer 2 velocities
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h_in_u, h2, p.Nx, p.Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h_in_v, h2, p.Nx, p.Ny)

    @. h_in_u = max(h_in_u, hmin)
    @. h_in_v = max(h_in_v, hmin)

    @. u2 = m2 / h_in_u
    @. v2 = n2 / h_in_v

    # Barotropic and heights (H on HGRID, Ubt on UGRID, Vbt on VGRID)
    Ubt = temp.temp_var_x1
    Vbt = temp.temp_var_y1

    eta = temp.temp_var_x2
    xi  = temp.temp_var_y2

    H = temp.temp_var_x3

    @. H = h1 + h2

    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_u!(h_in_u, H, p.Nx, p.Ny)
    @cuda threads=threads2 blocks=blocks2 k_recon_h_in_v!(h_in_v, H, p.Nx, p.Ny)

    @. h_in_u = max(h_in_u, hmin_bt)
    @. h_in_v = max(h_in_v, hmin_bt)

    @. Ubt = (m1 + m2) / h_in_u
    @. Vbt = (n1 + n2) / h_in_v

    η0 = FT(p.H1 + p.H2)
    @. eta = H - η0
    @. xi  = h2 - p.H2

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

    # Optional diagnostics output
    if nc.diag_fname != ""
        append_netcdf_diag(
            nc.diag_fname,
            nc.save_idx,
            nc.t,
            state.diag1,
            state.diag2,
        )
    end

    return NetcdfConfig(
        nc.fname,
        nc.diag_fname,
        nc.save_interval,
        nc.t,
        nc.save_idx + 1,
        nc.t + nc.save_interval,
        nc.do_save,
    )
end