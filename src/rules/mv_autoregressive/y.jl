@rule MAR(:y, Marginalisation) (m_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_Λ::Any, meta::MARMeta) =
begin
    ma, Va = mean_cov(q_a)
    mx, Wx = mean_invcov(m_x)

    mΛ = mean(q_Λ)

    order, ds = getorder(meta), getdimensionality(meta)

    mA = mar_companion_matrix(order, ds, ma)
    mW = mar_transition(getorder(meta), mΛ)
    dim = order*ds
    # this should be inside MARMeta
    es = [uvector(dim, i) for i in 1:order]
    Fs = [mask_mar(order, ds, i) for i in 1:order]
    
    Λ = sum(sum(es[j]'*mW*es[i]*Fs[j]*Va*Fs[i]' for i in 1:order) for j in 1:order)


    Ξ = Λ + Wx
    z = Wx*mx

    Vy = mA*inv(Ξ)*mA' + inv(Wx)
    my = mA*inv(Ξ)*z

    return MvNormalMeanCovariance(my, Vy)
end