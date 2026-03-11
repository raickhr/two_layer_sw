
function k_calc_curvature_terms!(
    curv_x, curv_y,  # output: curvature terms in hu- and hv-equations
    m, n, h,              # hu and hv at u and v points (previous timestep)
    u, v, 
    u_in_v, v_in_u, 
    nu_in_u, nu_in_v,
    buf_x, buf_y,
    lat_u, lat_v,              # latitudes of u and v points (degrees)
    Nx::Int, Ny::Int,
    earthRadius::FT,
)
    deg2rad = FT(π) / FT(180)

    adv_curv_x = buf_x
    adv_curv_y = buf_y

    @. adv_curv_x = h_in_u * u * v_in_u * tan(lat_u * deg2rad)/params.earthRadius
    @. adv_curv_y = -h_in_v * u_in_v^2   * tan(lat_v * deg2rad)/params.earthRadius

    @. curv_x =  FT(2) * h_in_u * nu_in_u * gradx_v_in_u * tan(lat_u * deg2rad)  / earthRadius
    @. curv_y = -FT(2) * h_in_v * nu_in_v * gradx_u_in_v * tan(lat_v * deg2rad)  / earthRadius
    

    
    return
end