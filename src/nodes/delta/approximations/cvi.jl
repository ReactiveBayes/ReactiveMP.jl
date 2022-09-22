export CVIApproximation
export renderCVI
export flux_update!
using Flux

struct CVIApproximation{R, O, F}
    n_samples::Int
    num_iterations::Int
    rng::R
    opt::O
    optupdate!::F
end

function CVIApproximation(n_samples, num_iterations, opt, optupdate!)
    return CVIApproximation(
        n_samples,
        num_iterations,
        nothing,
        opt,
        optupdate!
    )
end
#---------------------------
# CVI implementations
#---------------------------

function flux_update!(opt::O, λ::T, ∇::T) where {O, T <: NaturalParameters}
    return Flux.Optimise.update!(opt, vec(λ), vec(∇))
end

function renderCVI(logp_nc::Function,
    num_iterations::Int,
    opt,
    rng,
    λ_init::NormalNaturalParameters,
    msg_in::UnivariateGaussianDistributionsFamily,
    optupdate!)
    η = naturalparams(msg_in)
    λ = deepcopy(λ_init)

    df_m = (z) -> ForwardDiff.derivative(logp_nc, z)
    df_v = (z) -> 0.5 * ForwardDiff.derivative(df_m, z)
    rng = something(rng, Random.GLOBAL_RNG)

    for _ in 1:num_iterations
        q = convert(Distribution, λ)
        z_s = rand(rng, q)
        df_μ1 = df_m(z_s) - 2 * df_v(z_s) * mean(q)
        df_μ2 = df_v(z_s)
        ∇ = NormalNaturalParameters(
            λ.weighted_mean - η.weighted_mean - df_μ1,
            λ.minus_half_precision - η.minus_half_precision - df_μ2
        )
        λ_new = NormalNaturalParameters(optupdate!(opt, λ, ∇))
        if isproper(λ_new)
            λ = λ_new
        end
    end

    return λ
end

function renderCVI(logp_nc::Function,
    num_iterations::Int,
    opt::Any,
    rng::Any,
    λ_init::T,
    msg_in::Any,
    optupdate!) where {T <: NaturalParameters}
    η = naturalparams(msg_in)
    λ = deepcopy(λ_init)

    # convert lambda to vector
    # work within loop with vector

    A(vec_params) = lognormalizer(T(vec_params)) # maybe convert here makes more sense
    gradA(vec_params) = A'(vec_params) # Zygote
    Fisher(vec_params) = ForwardDiff.jacobian(gradA, vec_params) # Zygote throws mutating array error
    for _ in 1:num_iterations
        q = convert(Distribution, λ)
        _, q_friendly = logpdf_sample_friendly(q)

        if isnothing(rng)
            z_s = rand(q_friendly)
        else
            z_s = rand(rng, q_friendly)
        end
        logq(vec_params) = logpdf(T(vec_params), z_s)
        ∇logq = logq'(vec(λ))
        ∇f = Fisher(vec(λ)) \ (logp_nc(z_s) .* ∇logq)
        ∇ = λ - η - T(∇f)
        updated = T(optupdate!(opt, λ, ∇))
        if isproper(updated)
            λ = updated
        end
    end

    # convert vector result in parameters

    return λ
end

function renderCVI(logp_nc::Function,
    num_iterations::Int,
    opt,
    rng,
    λ_init::MvNormalNaturalParameters,
    msg_in::MultivariateGaussianDistributionsFamily,
    optupdate!)
    η = naturalparams(msg_in)
    λ = deepcopy(λ_init)

    df_m = (z) -> ForwardDiff.gradient(logp_nc, z)
    df_v = (z) -> 0.5 * ForwardDiff.jacobian(df_m, z)

    for _ in 1:num_iterations
        q = convert(Distribution, λ)

        _, q_friendly = logpdf_sample_friendly(q)

        if isnothing(rng)
            z_s = rand(q_friendly) # need to add rng here (or maybe better to do callback)
        else
            z_s = rand(rng, q_friendly)
        end

        df_μ1 = df_m(z_s) - 2 * df_v(z_s) * mean(q)
        df_μ2 = df_v(z_s)
        ∇f = MvNormalNaturalParameters([df_μ1; vec(df_μ2)])

        ∇ = λ - η - ∇f

        updated = MvNormalNaturalParameters(optupdate!(opt, λ, ∇))

        if isproper(updated)
            λ = updated
        end
    end

    return λ
end

## DeltaFn node non-standard rule layout 

struct CVIApproximationDeltaFnRuleLayout end

deltafn_rule_layout(::DeltaFnNode, ::CVIApproximation) = CVIApproximationDeltaFnRuleLayout()

deltafn_apply_layout(::CVIApproximationDeltaFnRuleLayout, ::Val{:q_out}, model, factornode::DeltaFnNode) =
    deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:q_out), model, factornode)

deltafn_apply_layout(::CVIApproximationDeltaFnRuleLayout, ::Val{:q_ins}, model, factornode::DeltaFnNode) =
    deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:q_ins), model, factornode)

deltafn_apply_layout(::CVIApproximationDeltaFnRuleLayout, ::Val{:m_in}, model, factornode::DeltaFnNode) =
    deltafn_apply_layout(DeltaFnDefaultRuleLayout(), Val(:m_in), model, factornode)

# This function declares how to compute `m_out` 
function deltafn_apply_layout(::CVIApproximationDeltaFnRuleLayout, ::Val{:m_out}, model, factornode::DeltaFnNode)
    let interface = factornode.out
        # By default, CVI does not need an inbound message 
        msgs_names      = nothing
        msgs_observable = of(nothing)

        # By default, CVI requires `q_ins`
        marginal_names       = Val{(:ins,)}
        marginals_observable = combineLatestUpdates((getstream(factornode.localmarginals.marginals[2]),), PushNew())

        fform       = functionalform(factornode)
        vtag        = tag(interface)
        vconstraint = local_constraint(interface)
        meta        = metadata(factornode)

        vmessageout = combineLatest((msgs_observable, marginals_observable), PushNew())
        # TODO
        # vmessageout = apply_pipeline_stage(get_pipeline_stages(interface), factornode, vtag, vmessageout)

        mapping =
            let messagemap = MessageMapping(fform, vtag, vconstraint, msgs_names, marginal_names, meta, factornode)
                (dependencies) -> VariationalMessage(dependencies[1], dependencies[2], messagemap)
            end

        vmessageout = vmessageout |> map(AbstractMessage, mapping)
        vmessageout = apply_pipeline_stage(get_pipeline_stages(getoptions(model)), factornode, vtag, vmessageout)
        # TODO
        # vmessageout = apply_pipeline_stage(node_pipeline_extra_stages, factornode, vtag, vmessageout)
        vmessageout = vmessageout |> schedule_on(global_reactive_scheduler(getoptions(model)))

        connect!(messageout(interface), vmessageout)
    end
end
