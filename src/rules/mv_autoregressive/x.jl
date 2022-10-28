
@rule MAR(:x, Marginalisation) (m_y::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta) =
begin
    ma, Va = mean_cov(q_a)
    my, Vy = mean_cov(m_y)

    mΛ = mean(q_Λ)

    order, ds = getorder(meta), getdimensionality(meta)
    dim = order*ds

    mA = mar_companion_matrix(order, ds, ma)
    mW = mar_transition(getorder(meta), mΛ)
    
    # this should be inside MARMeta
    es = [uvector(dim, i) for i in 1:order]
    Fs = [mask_mar(order, ds, i) for i in 1:order]

    # ∏ = Iterators.product(transpose.(es), mW, es, Fs, Va, transpose.(Fs))
    Σ = sum(sum(es[j]'*mW*es[i]*Fs[j]*Va*Fs[i]' for i in 1:order) for j in 1:order)

    Ξ = mA'*inv(Vy + inv(mW))*mA + inv(Σ)
    z = mA'*inv(Vy + inv(mW))*my

    mx = inv(Ξ)*z
    Vx = inv(Ξ)

    return MvNormalMeanCovariance(mx, Vx)
end