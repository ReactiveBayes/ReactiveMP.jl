export CVIApproximation

using Random

"""
    cvi_update!(opt, λ, ∇)
"""
function cvi_update! end

function cvi_update!(callback::F, λ, ∇) where {F <: Function}
    return callback(λ, ∇)
end

cvilinearize(vector::AbstractVector) = vector
cvilinearize(matrix::AbstractMatrix) = eachcol(matrix)

"""
CVIApproximation

To call the delta node with CVI method, you need to specify its meta to CVIApproximation.
CVIApproximation object is a container for 4 CVI parameters:
    rng - random number generator
    n_samples - number of samples for our rule
    num_iterations - number of iteration inside renderCVI
    opt - optimizer, which will be used for iteration call inside render CVI

CVIApproximation has two constructors, one with all parameters specified and one where rng is missed (in this case, the global rng will be used in its place):
    `CVIApproximation(rng, number of out sample, number of iterations inside render cvi, optimizer)`: all parameters specified
    `CVIApproximation(number of out samples, number of iterations inside render cvi, optimizer)`: you will use global rng
"""
struct CVIApproximation{R, O}
    rng::R
    n_samples::Int
    num_iterations::Int
    opt::O
end

function CVIApproximation(n_samples::Int, num_iterations::Int, opt::O) where {O}
    return CVIApproximation(Random.GLOBAL_RNG, n_samples, num_iterations, opt)
end

"""
Alias for the `CVIApproximation` method.

See also: [`CVIApproximation`](@ref)
"""
const CVI = CVIApproximation

#---------------------------
# CVI implementations
#---------------------------

get_df_m(
    ::Type{<:UnivariateNormalNaturalParameters},
    ::Type{<:UnivariateGaussianDistributionsFamily},
    logp_nc::Function
) = (z) -> ForwardDiff.derivative(logp_nc, z)

get_df_m(
    ::Type{<:MvNormalNaturalParameters},
    ::Type{<:MultivariateGaussianDistributionsFamily},
    logp_nc::Function
) = (z) -> ForwardDiff.gradient(logp_nc, z)

get_df_v(
    ::Type{<:UnivariateNormalNaturalParameters},
    ::Type{<:UnivariateGaussianDistributionsFamily},
    df_m::Function
) = (z) -> ForwardDiff.derivative(df_m, z)

get_df_v(::Type{<:MvNormalNaturalParameters}, ::Type{<:MultivariateGaussianDistributionsFamily}, df_m::Function) =
    (z) -> ForwardDiff.jacobian(df_m, z)

function renderCVI(logp_nc::Function,
    num_iterations::Int,
    opt,
    rng,
    λ_init::T,
    msg_in::GaussianDistributionsFamily) where {T <: NormalNaturalParameters}
    η = naturalparams(msg_in)
    λ = deepcopy(λ_init)

    df_m = (z) -> get_df_m(typeof(λ_init), typeof(msg_in), logp_nc)(z)
    df_v = (z) -> 0.5 * get_df_v(typeof(λ_init), typeof(msg_in), df_m)(z)
    rng = something(rng, Random.GLOBAL_RNG)

    for _ in 1:num_iterations
        q = convert(Distribution, λ)
        z_s = rand(rng, q)
        df_μ1 = df_m(z_s) - 2 * df_v(z_s) * mean(q)
        df_μ2 = df_v(z_s)
        ∇f = T(df_μ1, df_μ2)
        ∇ = λ - η - ∇f
        λ_new = T(cvi_update!(opt, λ, ∇))
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
    msg_in::Any) where {T <: NaturalParameters}
    η = naturalparams(msg_in)
    λ = deepcopy(λ_init)

    # convert lambda to vector
    # work within loop with vector
    rng = something(rng, Random.GLOBAL_RNG)

    A = (vec_params) -> lognormalizer(T(vec_params)) # maybe convert here makes more sense
    gradA = (vec_params) -> ForwardDiff.gradient(A, vec_params)
    Fisher = (vec_params) -> ForwardDiff.jacobian(gradA, vec_params)

    for _ in 1:num_iterations
        q = convert(Distribution, λ)
        _, q_friendly = logpdf_sample_friendly(q)

        z_s = rand(rng, q_friendly)

        logq = (vec_params) -> logpdf(T(vec_params), z_s)
        ∇logq = ForwardDiff.gradient(logq, vec(λ))

        ∇f = Fisher(vec(λ)) \ (logp_nc(z_s) .* ∇logq)
        ∇ = λ - η - T(∇f)
        updated = T(cvi_update!(opt, λ, ∇))
        if isproper(updated)
            λ = updated
        end
    end

    # convert vector result in parameters

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
