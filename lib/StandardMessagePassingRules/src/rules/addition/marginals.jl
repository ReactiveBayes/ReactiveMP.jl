# The joint of the inputs: with a normal message on `out`, the Gaussian inputs jointly and each
# known one on its own; with `out` known, the inputs on the plane where they sum to it.
@define_marginal_update_rule(
    node = +, target = (:in,),
    args = (m[:out]::NormalDistributionsFamily, m[:in...]::NormalOrPoint),
    body = (args) -> sum_inputs_joint(args.m[:out], args.m[:in]),
)

@define_marginal_update_rule(
    node = +, target = (:in,),
    args = (m[:out]::PointMass, m[:in...]::NormalOrPoint),
    body = (args) -> sum_inputs_joint(args.m[:out], args.m[:in]),
)
