@rule MAR(:Λ, Marginalisation) (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_a::MultivariateNormalDistributionsFamily,
    meta::MARMeta
) = begin
    order, ds = getorder(meta), getdimensionality(meta)
    F = Multivariate
    dim = order*ds

    n = div(ndims(q_y_x), 2)

    y_x_mean, y_x_cov = mean_cov(q_y_x)
    ma, Va = mean_cov(q_a)

    mA = mar_companion_matrix(order, ds, ma)

    myx, Vyx = mean_cov(q_y_x)
    my, Vy   = ar_slice(F, myx, 1:dim), ar_slice(F, Vyx, 1:dim, 1:dim)
    mx, Vx   = ar_slice(F, myx, (dim+1):2dim), ar_slice(F, Vyx, (dim+1):2dim, (dim+1):2dim)
    Vyx      = ar_slice(F, Vyx, (dim+1):2dim, 1:dim)

    # this should be inside MARMeta
    es = [uvector(dim, i) for i in 1:order]
    Fs = [mask_mar(order, ds, i) for i in 1:order]

    vmx = (Vx + mx*mx')
    S = mar_shift(order, ds)
    G₁ = S*vmx*S'
    G₂ = sum(es[i]*ma'Fs[i]'vmx for i in 1:order)*S'
    G₃ = transpose(G₂)
    G₄ = sum(sum(es[i]*ma'Fs[i]'*vmx*Fs[j]*ma*es[j]' + es[i]*tr(Va*Fs[i]'*vmx*Fs[j])*es[j]' for i in 1:order) for j in 1:order)
    G = G₁ + G₂ + G₃ + G₄
    
    @show Δ = G + Vy + my*my' - (Vyx + my*mx')*mA' - mA*(Vyx'+ mx*my')

    return WishartMessage(n-2, Δ)
end