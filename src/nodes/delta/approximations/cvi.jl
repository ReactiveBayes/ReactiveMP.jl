export CVIApproximation
export renderCVI

struct CVIApproximation
    learning_rate
    n_samples
    rng
    opt
    dataset_size
    batch_size
end

#---------------------------
# CVI implementations
#---------------------------

function renderCVI(logp_nc::Function,
    num_iterations::Int,
    opt,
    λ_init::NormalNaturalParametrs,
    msg_in::UnivariateGaussianDistributionsFamily)
    η = naturalParams(msg_in)
    λ = deepcopy(λ_init)

    df_m(z) = ForwardDiff.derivative(logp_nc, z)
    df_v(z) = 0.5 * ForwardDiff.derivative(df_m, z)

    for _ in 1:num_iterations
        q = standardDist(λ)
        z_s = sample(q) # need to add rng here parameter
        df_μ1 = df_m(z_s) - 2 * df_v(z_s) * mean(q)
        df_μ2 = df_v(z_s)
        ∇f = NormalNaturalParametrs(df_μ1, df_μ2)

        # λ_old = deepcopy(λ)

        ∇ = λ - η - ∇f
        update!(opt, λ, ∇)

        # if isProper(standardDist(λ)) == false
        #     λ = λ_old
        # end
    end

    return λ
end
