@define_message_update_rule(
    node = dot, target = :in2, ctx = (:matrix_correction,),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:in1]::PointMass),
    body = (ctx, args) -> dot_backward(ctx, args.m[:in1], args.m[:out]),
)

@define_message_update_rule(
    node = dot, target = :in2,
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:in1]::NormalDistributionsFamily),
    body = (args) -> error(DOT_OF_NORMALS),
)
