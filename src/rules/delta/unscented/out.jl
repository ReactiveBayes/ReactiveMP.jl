import Distributions: LogNormal
@rule DeltaFn(:out, Marginalisation) (m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {N, M <: Unscented} = begin
    return approximate(getmethod(meta), getnodefn(meta, Val(:out)), m_ins)
end

# ##### Test gp, only for exponential function 
@rule DeltaFn(:out, Marginalisation) (m_ins::ManyOf{1, GaussianProcess}, meta::Tuple{ProcessMeta, DeltaMeta}) = begin 
    gp_finitemarginal = m_ins[1].finitemarginal
    m_gp, cov_gp = mean_cov(gp_finitemarginal)
    index = meta[1].index
    kernelf = m_ins[1].kernelfunction
    test    = m_ins[1].testinput
    meanf   = m_ins[1].meanfunction
    train   = m_ins[1].traininput
    cov_strategy = m_ins[1].covariance_strategy
    x_u = m_ins[1].inducing_input

    mean_gp, var_gp = ReactiveMP.predictMVN(cov_strategy, kernelf,meanf,test,[train[index]],m_gp, x_u) #changed here

    # return approximate(getmethod(meta[2]), getnodefn(ReactiveMP.Val(:out)), (mean_gp,), (var_gp,))
    #only use this for exponential function 
    return LogNormal(mean_gp[], var_gp[])
end