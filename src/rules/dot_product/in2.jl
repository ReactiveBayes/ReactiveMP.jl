
@rule typeof(dot)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass, meta::AbstractCorrection) = begin
    n = length(mean(m_in1))
    m = zeros(n)
    P = diageye(n)
    dist = MvNormalMeanCovariance(m,P)
    g = (x) -> exp(logpdf(m_out,dot(mean(m_in1),x)) - logpdf(dist,x))

    method = ghcubature(9)
    weights = getweights(method, m, P)
    points  = getpoints(method, m, P)
    Z = 0

    for (index, (weight, point)) in enumerate(zip(weights, points))
        gv = g(point)
        cv = weight * gv
        Z += cv
    end
    if isnothing(messages[1])
        @logscale log(Z)
    else
        @logscale log(Z) + getlogscale(messages[1])#correct 
    end

    A = mean(m_in1)
    out_wmean, out_prec = weightedmean_precision(m_out)

    ξ = A * out_wmean
    W = correction!(meta, v_a_vT(A, out_prec))

    return convert(promote_variate_type(variate_form(m_in1), NormalWeightedMeanPrecision), ξ, W)
end
