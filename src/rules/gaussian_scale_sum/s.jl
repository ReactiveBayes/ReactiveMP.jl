export rule

@rule GaussianScaleSum(:s, Marginalisation) (q_out::Any, q_n::NormalDistributionsFamily) = begin
    
    # fetch parameters
    mx, vx = mean(q_out), cov(q_out)
    mn = mean(q_n)

    # calculate outward message
    f = let mn = mn, mx = mx, vx = vx
        s -> -log(exp(s) + exp(mn)) - (vx + abs2(mx))/ (exp(s) + exp(mn))
    end

    return NormalLikelihood(f, LaplaceApproximation())
end
