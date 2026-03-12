function run_twoLayer_SW(; kwargs...)
# Driver-level defaults (not stored in Params)
    driver_defaults = (
        save_interval = 3600.0,
        end_time      = 24*3600.0,
        out_netcdf    = "two_layer_SW.nc",
    )
    kw = merge(driver_defaults, kwargs)

    # 1) Params, grid, state
    p     = make_params(; kw...)          # passes overrides too
    grid  = build_gridVars(p)
    state = allocate_state(p)

    # Wind IC
    build_analytical_wind!(state, p)

    # NetCDF config (use kw.*)
    netcfg = setup_netcdf_if_needed!(
        p, grid,
        kw.out_netcdf,
        kw.diag_netcdf,
        kw.save_interval,
    )

    # Threads/blocks
    threads2 = (16, 16)
    blocks2  = (cld(p.Nx, threads2[1]), cld(p.Ny, threads2[2]))

    threads1 = 128
    blocks1  = cld(p.Nx, threads1)

    nsteps   = Int(ceil(kw.end_time / p.dt))
    prog_bar = Progress(nsteps; dt = 0.0, desc = "Time stepping")

    # Initial condition (Kelvin wave bulb)
    initialize_surface_bulb!(state, p, 
                             η0   = FT(0.15),
                             R    = FT(1000e3),
                             lon0 = FT(p.lon1 + p.Nx/FT(4.0) * p.dlon),
                             lat0 = FT(p.lat1 + p.Ny/FT(2.0) * p.dlat))

    # Inital baroclinic jet 
    # initialize_baroclinic_tanh_jet!(state, p, Δξ = FT(400.0), L = FT(100.0e3), noise_amp = FT(0.05))

    # 4. Main loop
    current_time = 0.0
    # write IC
    netcfg = write_state!(netcfg, current_time, state, p, threads2, blocks2)

    step = 0
    while current_time < kw[:end_time]
        step += 1
        # ---- Time bookkeeping ----
        current_time += p.dt

        # ---- Forcing ----
        update_surface_and_bottom_stress!(
            state, p;
            threads2 = threads2, blocks2 = blocks2,
        )

        # ---- Baroclinic part ----
        step_baroclinic!( 
            state, grid, p;
            threads1 = threads1, blocks1 = blocks1,
            threads2 = threads2, blocks2 = blocks2,
            step = step,
        )
        
        # ---- NetCDF output (if enabled & on schedule) ----
        netcfg = write_state!(netcfg, current_time, state, p, threads2, blocks2)

        next!(prog_bar; showvalues = [(:time_days, round(current_time / 86400, digits = 4))])
    end

    return nothing
end
