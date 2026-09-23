@define_message_update_rule(
    node = dot, target = :in1, ctx = (:matrix_correction,),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:in2]::PointMass),
    body = (ctx, args) -> dot_backward(ctx, args.m[:in2], args.m[:out]),
)

@define_message_update_rule(
    node = dot, target = :in1,
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:in2]::NormalDistributionsFamily),
    body = (args) -> error(DOT_OF_NORMALS),
)
