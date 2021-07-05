export rule

@rule DenseReLU(:z, Marginalisation) (q_output::PointMass, q_input::PointMass, q_w::NormalMeanPrecision, q_f::NormalMeanPrecision, meta::DenseReLUMeta) = begin
    
    # extract required statistics
    mf = mean(q_f)

    # extract parameters
    C = getC(meta)

    # calculate new statistics
    p = sigmoid(C * mf)

    # return message
    return Bernoulli(p)

end

sigmoid(x) = 1/(1+exp(-x))