export rule

@rule DenseReLU(:w, Marginalisation) (q_output::PointMass, q_input::PointMass, q_z::Bernoulli, q_f::NormalMeanPrecision, meta::DenseReLUMeta) = begin
    
    # extract required statistics
    mf = mean(q_f)
    mx = mean(q_input)

    # extract parameters
    β = getβ(meta)

    # calculate new statistics
    mw = mf/mx
    ww = β*mx^2

    # return message
    return GaussianMeanPrecision(mw, ww)

end