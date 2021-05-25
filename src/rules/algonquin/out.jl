export rule

@rule Algonquin(:out, Marginalisation) (q_s::NormalDistributionsFamily, q_n::NormalDistributionsFamily, q_γ::PointMass) = begin
    
    # fetch parameters
    ms = mean(q_s)
    mn = mean(q_n)
    γ = mean(q_γ)

    # calculate new parameters
    mout = log(exp(ms)+exp(mn))
    wout = γ

    return NormalMeanPrecision(mout, wout)
end
