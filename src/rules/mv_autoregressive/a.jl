
@rule MAR(:a, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta) = begin
    
    order, ds = getorder(meta), getdimensionality(meta)
    F = Multivariate

    dim = order*ds

    myx, Vyx = mean_cov(q_y_x)
    my, Vy   = ar_slice(F, myx, 1:dim), ar_slice(F, Vyx, 1:dim, 1:dim)
    mx, Vx   = ar_slice(F, myx, (dim+1):2dim), ar_slice(F, Vyx, (dim+1):2dim, (dim+1):2dim)
    Vyx      = ar_slice(F, Vyx, (dim+1):2dim, 1:dim)
    mΛ = mean(q_Λ)
    mW = mar_transition(order, mΛ)
    mW
    # this should be inside MARMeta
    es = [uvector(dim, i) for i in 1:order]
    Fs = [mask_mar(order, ds, i) for i in 1:order]
    # @show Iterators.product(transpose.(es), mW, es)
    # @show sum(prod, Iterators.product(transpose.(es), mW, es))
    # ∏ = Iterators.product(transpose.(es), mW, es, transpose.(Fs), (Vx + mx*mx'), Fs)

    D = sum(sum(es[i]'*mW*es[j]*Fs[i]'*(Vx + mx*mx')*Fs[j] for i in 1:order) for j in 1:order)
    z = sum(Fs[i]'*(Vyx + my*mx')*mW*es[i] for i in 1:order)
    return MvNormalMeanCovariance(inv(D)*z, inv(D))
end