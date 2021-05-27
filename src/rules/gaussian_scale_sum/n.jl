export rule

@rule GaussianScaleSum(:n, Marginalisation) (q_out::Any, q_s::NormalDistributionsFamily) = begin
    
    # fetch parameters
    mx, vx = mean(q_out), cov(q_out)
    ms = mean(q_s)

    # calculate outward message
    f = let ms = ms, mx = mx, vx = vx
        n -> -log(exp(ms) + exp(n)) - (real(vx) + abs2(mx))/ (exp(ms) + exp(n))
    end

    return NormalLikelihood(f, LaplaceApproximation())
end
