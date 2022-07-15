export CVIApproximation
export renderCVI
using Flux

mutable struct CVIApproximation
    n_samples
    num_iterations
    rng
    opt
    dataset_size
    batch_size
end

function CVIApproximation(n_samples, num_iterations, opt, dataset_size, batch_size)
    return CVIApproximation(
        n_samples,
        num_iterations,
        nothing,
        opt,
        dataset_size,
        batch_size
    )
end
#---------------------------
# CVI implementations
#---------------------------

function renderCVI(logp_nc::Function,
    num_iterations::Int,
    opt,
    rng,
    λ_init::NormalNaturalParametrs,
    msg_in::UnivariateGaussianDistributionsFamily)
    η = naturalParams(msg_in)
    λ = deepcopy(λ_init)

    df_m(z) = ForwardDiff.derivative(logp_nc, z)
    df_v(z) = 0.5 * ForwardDiff.derivative(df_m, z)

    for _ in 1:num_iterations
        q = standardDist(λ)

        if isnothing(rng)
            z_s = rand(q) # need to add rng here (or maybe better to do callback)
        else
            z_s = rand(rng, q)
        end

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
    rng::Any,
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

        if isnothing(rng)
            z_s = rand(q_friendly)
        else
            z_s = rand(rng, q_friendly)
        end

        println("sampling is okay")

        println(λ)
        println(z_s)
        logq(vec_params) = logPdf(T(vec_params), z_s)

        ∇logq = logq'(vec(λ))
        print("logq is okay")

        ∇f = Fisher(vec(λ)) \ (logp_nc(z_s) .* ∇logq)
        println("div is okay")

        ∇ = λ - η - T(∇f)

        println("update is okay")

        updated = T(Flux.Optimise.update!(opt, vec(λ), vec(∇)))
        if isProper(updated)
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
    λ_init::MvNormalNaturalParametrs,
    msg_in::MultivariateGaussianDistributionsFamily)
    η = naturalParams(msg_in)
    λ = deepcopy(λ_init)

    df_m(z) = ForwardDiff.gradient(logp_nc, z)
    df_v(z) = 0.5 * ForwardDiff.jacobian(df_m, z)

    for _ in 1:num_iterations
        q = standardDist(λ)

        _, q_friendly = logpdf_sample_friendly(q)

        if isnothing(rng)
            z_s = rand(q_friendly) # need to add rng here (or maybe better to do callback)
        else
            z_s = rand(rng, q_friendly)
        end

        df_μ1 = df_m(z_s) - 2 * df_v(z_s) * mean(q)
        df_μ2 = df_v(z_s)
        ∇f = MvNormalNaturalParametrs([df_μ1; vec(df_μ2)])

        ∇ = λ - η - ∇f

        updated = MvNormalNaturalParametrs(Flux.Optimise.update!(opt, vec(λ), vec(∇)))

        if isProper(updated)
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
