export CVIApproximation
export renderCVI
using Flux

mutable struct CVIApproximation
    learning_rate
    n_samples
    num_iterations
    rng
    opt
    dataset_size
    batch_size
    q_ins_marginal
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
        z_s = rand(q) # need to add rng here (or maybe better to do callback)
        df_μ1 = df_m(z_s) - 2 * df_v(z_s) * mean(q)
        df_μ2 = df_v(z_s)
        ∇f = NormalNaturalParametrs(df_μ1, df_μ2)
        ∇ = λ - η - ∇f
        λ = NormalNaturalParametrs(Flux.Optimise.update!(opt, vec(λ), vec(∇)))
    end

    return λ
end

function renderCVI(logp_nc::Function,
    num_iterations::Int,
    opt::Any,
    λ_init::T,
    msg_in::Any) where {T <: NaturalParametrs}
    η = naturalParams(msg_in)
    λ = deepcopy(λ_init)

    # convert lambda to vector
    # work within loop with vector

    A(vec_params) = logNormalizer(T(vec_params)) # maybe convert here makes more sense
    gradA(vec_params) = A'(vec_params) # Zygote
    Fisher(vec_params) = ForwardDiff.jacobian(gradA, vec_params) # Zygote throws mutating array error
    for _ in 1:num_iterations
        q = standardDist(λ)
        _, q_friendly = logpdf_sample_friendly(q)
        z_s = rand(q_friendly) # use rng from CVI meta

        logq(vec_params) = logPdf(T(vec_params), z_s)

        ∇logq = logq'(vec(λ))
        ∇f = Fisher(vec(λ)) \ (logp_nc(z_s) .* ∇logq)
        ∇ = λ - η - T(∇f)
        updated = T(Flux.Optimise.update!(opt, vec(λ), vec(∇)))
        if isProper(updated)
            λ = updated
        end
    end

    # convert vector result in parameters

    return λ
end
