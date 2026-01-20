function k_add_coriolisforce!(
    m_new, n_new,  # hu and hv in u and v points after coriolis application
    m_old, n_old,  # hu and hv in u and v points of previous timestep
    m_star, n_star, # intermediate hu and hv in u and v points due to all terms but coriolis
    lat_u, lat_v,  # latitudes of u and v points
    Nx::Int, Ny::Int, dt::FT, Ω::FT)
    """
    This function adds coriolis force 
    """
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if i <= Nx && j <= Ny
        im1 = iper(i-1, Nx)
        ip1 = iper(i+1, Nx)
        
        jm1 = clamp1(j-1, Ny)
        jp1 = clamp1(j+1, Ny)
        
        α_u = dt * coriolis_at_lat(Ω, lat_u[j])
        α_v = dt * coriolis_at_lat(Ω, lat_v[j])
        β_u = FT(0.25)*α_u^2  
        β_v = FT(0.25)*α_v^2  

        """
        Reconstruction
            Cell indexing is as follows. Below is one single cell for h
                |----------------v(i,j)----------------|
                |                                      |
                |                                      |
                |                                      |
             u(i-1,j)            h(i,j)               u(i,j)
                |                                      |
                |                                      |
                |                                      |
                |-------------- v(i, j-1)--------------|
         """

        # Reconstruct m in v points
        m_old_in_v  = FT(0.25) * (m_old[i,j]  + m_old[i,jp1] +
                                  m_old[im1,j] + m_old[im1,jp1]) 

        # Reconstruct n in u points
        n_old_in_u  = FT(0.25) * (n_old[i,j]   + n_old[ip1,j] +
                                  n_old[i,jm1] + n_old[ip1,jm1])
                                  
        
        m_new[i, j] = (m_star[i,j] - β_u * m_old[i, j] + α_u * n_old_in_u )/(1 + β_u)
        n_new[i, j] = (n_star[i,j] - β_v * n_old[i, j] - α_v * m_old_in_v )/(1 + β_v)
    end
    return
end


function k_calc_curvature_terms!(
    curvature_x, curvature_y,  # curvature terms, output of the function
    m_old, n_old,  # hu and hv in u and v points of previous timestep
    h_old_in_u, h_old_in_v, # h in u and v points of previous time step
    lat_u, lat_v,  # latitudes of u and v points
    Nx::Int, Ny::Int, earthRadius::FT)
    """
    This function adds coriolis force 
    """
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    deg2rad = FT(π)/FT(180)
    if i <= Nx && j <= Ny
        im1 = iper(i-1, Nx)
        ip1 = iper(i+1, Nx)
        
        jm1 = clamp1(j-1, Ny)
        jp1 = clamp1(j+1, Ny)

        """
        Reconstruction
            Cell indexing is as follows. Below is one single cell for h
                |----------------v(i,j)----------------|
                |                                      |
                |                                      |
                |                                      |
             u(i-1,j)            h(i,j)               u(i,j)
                |                                      |
                |                                      |
                |                                      |
                |-------------- v(i, j-1)--------------|
         """

        # Reconstruct m in v points
        m_old_in_v  = FT(0.25) * (m_old[i,j]  + m_old[i,jp1] +
                                  m_old[im1,j] + m_old[im1,jp1]) 
        u_old_in_v = m_old_in_v/h_old_in_v[i,j]
        

        # Reconstruct n in u points
        n_old_in_u  = FT(0.25) * (n_old[i,j]   + n_old[ip1,j] +
                                  n_old[i,jm1] + n_old[ip1,jm1])
        v_old_in_u = n_old_in_u/h_old_in_u[i,j]

        # u in native location
        u_old = m_old[i, j]/h_old_in_u[i,j]
        
        #curvature_x = - u v tan(φ)/R
        curvature_x[i, j] = -h_old_in_u[i, j] * u_old * v_old_in_u * tan(lat_u[j] * deg2rad) / earthRadius

        #curvature_x =  u² tan(φ)/R
        curvature_y[i, j] = h_old_in_v[i, j] * u_old_in_v * u_old_in_v * tan(lat_v[j] * deg2rad)/ earthRadius
    end
    return
end