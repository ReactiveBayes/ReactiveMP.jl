
@rule Sigmoid(:in, Marginalisation) (q_out::Categorical, q_ξ::PointMass) = begin

    mout = mean(q_out)
    ξ_hat = mean(q_ξ)
    w = 2 * ((logistic(ξ_hat) - 0.5)/(2*ξ_hat))
    mout_w = (mout - 0.5)* w
    return NormalWeightedMeanPrecision(mout_w, w)  
end