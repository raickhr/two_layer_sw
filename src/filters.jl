function k_apply_shapiro_filter!(
    H,        # after applying Shapiro filter
    H_star,   # before applying Shapiro filter
    smoothing_eps::FT,
    Nx::Int, Ny::Int)
    """
    Apply a Shapiro-like smoother to the free surface:
    H_new = (1-ε) H + ε * (mean of 4 neighbors)
    """
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= Nx && j <= Ny
        im1 = iper(i-1, Nx)
        ip1 = iper(i+1, Nx)

        jm1 = clamp1(j-1, Ny)
        jp1 = clamp1(j+1, Ny)

        center = H_star[i,j]
        neigh_sum = H_star[im1,j] + H_star[ip1,j] +
                    H_star[i,jm1] + H_star[i,jp1]

        H[i,j] = (FT(1.0) - smoothing_eps) * center +
                 FT(0.25) * smoothing_eps * neigh_sum
    end
    return
end