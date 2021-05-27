export rule

@rule GaussianScaleSum(:out, Marginalisation) (q_s::NormalDistributionsFamily, q_n::NormalDistributionsFamily) = begin
    
    # fetch parameters
    ms = mean(q_s)
    mn = mean(q_n)

    # calculate new parameters
    vout = exp(ms)+exp(mn)

    # return distribution
    return ComplexNormal(0, vout, 0)
end
