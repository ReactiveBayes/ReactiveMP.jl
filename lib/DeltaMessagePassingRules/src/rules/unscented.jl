# The Unscented rules, ported from v6's `rules/delta/unscented/`, most of whose routines came
# from ForneyLab.jl.

@define_message_update_rule(
    node = DeltaFn, target = :out, algorithm = DeltaApproximation{<:Unscented}, ctx = (:node,),
    args = (m[:in...]::NormalDistributionsFamily,),
    body = (algo, ctx, args) -> approximate_normal(algo.method, getnodefn(ctx.node, Target(:out)), args.m[:in]),
)

# The joint over the inputs: the forward statistics through the function, smoothed with the
# message from `out` (RTS, Petersen et al. 2018).
@define_marginal_update_rule(
    node = DeltaFn, target = (:in,), algorithm = DeltaApproximation{<:Unscented}, ctx = (:node,),
    args = (m[:out]::NormalDistributionsFamily, m[:in...]::NormalDistributionsFamily),
    body = (algo, ctx, args) -> begin
        inputs = args.m[:in]
        statistics = mean_cov.(inputs)
        μs_fw_in, Σs_fw_in = first.(statistics), last.(statistics)
        μ_tilde, Σ_tilde, C_tilde = unscented_statistics(algo.method, getnodefn(ctx.node, Target(:out)), μs_fw_in, Σs_fw_in)
        joint = convert(JointNormal, μs_fw_in, Σs_fw_in)
        μ_fw_in, Σ_fw_in = mean_cov(joint)
        μ_bw_out, Σ_bw_out = mean_cov(args.m[:out])
        μ_in, Σ_in = smoothRTS(μ_tilde, Σ_tilde, C_tilde, μ_fw_in, Σ_fw_in, μ_bw_out, Σ_bw_out)
        JointNormal(convert(promote_variate_type(typeof(μ_in), NormalMeanVariance), μ_in, Σ_in), size.(inputs))
    end,
)

# Without an inverse: the input's share of the joint, divided by its own message.
@define_message_update_rule(
    node = DeltaFn, target = (:in, k), algorithm = DeltaApproximation{<:Unscented, Nothing},
    args = (m[:in][k]::NormalDistributionsFamily, q[(:in,)]::JointNormal),
    body = (args) -> begin
        ξ_in, Λ_in = weightedmean_precision(component(args.q[(:in,)], k))
        ξ_fw, Λ_fw = weightedmean_precision(args.m[:in][k])
        # The subtraction may leave a precision that is not positive definite, as in v6.
        convert(promote_variate_type(typeof(ξ_in), NormalWeightedMeanPrecision), ξ_in - ξ_fw, Λ_in - Λ_fw)
    end,
)

# With an inverse: the message from `out` and the other inputs' messages, through it.
@define_message_update_rule(
    node = DeltaFn, target = (:in, k), algorithm = DeltaApproximation{<:Unscented, <:Union{Function, Tuple{Vararg{Function}}}},
    args = (m[:out]::NormalDistributionsFamily, m[:in][!k]::NormalDistributionsFamily),
    body = (algo, args) -> approximate_normal(algo.method, inverse_towards(algo, k), (args.m[:out], filter(!isnothing, args.m[:in])...)),
)
