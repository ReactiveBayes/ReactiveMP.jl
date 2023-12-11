@rule ContinuousTransition(:a, Marginalisation) (q_y_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Any, meta::CTMeta) = begin
    ma = mean(q_a)
    mW = mean(q_W)
    myx, Vyx = mean_cov(q_y_x)

    Fs = getjacobians(meta, ma)
    dy = length(Fs)

    mx, Vx = @views myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = @views myx[1:dy], Vyx[1:dy, 1:dy]
    Vyx    = @view Vyx[1:dy, (dy + 1):end]

    # rank1update(Vyx, mx, my) equivalent to ξ = (Vyx + mx * my') 
    D = sum(sum(StandardBasisVector(dy, j)' * mW * StandardBasisVector(dy, i) * Fs[i]' * rank1update(Vx, mx) * Fs[j] for i in 1:dy) for j in 1:dy)
    z = sum(Fs[i]' * rank1update(Vyx', mx, my) * mW * StandardBasisVector(dy, i) for i in 1:dy)

    return MvNormalWeightedMeanPrecision(z, D)
end
