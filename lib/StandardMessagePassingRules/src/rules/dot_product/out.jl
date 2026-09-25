# A known vector's dot product with a Gaussian is its pushforward, normalised: log scale 0.
@define_message_update_rule(node = dot, target = :out, args = (m[:in1]::PointMass, m[:in2]::NormalDistributionsFamily), logscale = 0, body = (args) -> dot_forward(mean(args.m[:in1]), args.m[:in2]))

@define_message_update_rule(node = dot, target = :out, args = (m[:in1]::NormalDistributionsFamily, m[:in2]::PointMass), logscale = 0, body = (args) -> dot_forward(mean(args.m[:in2]), args.m[:in1]))

@define_message_update_rule(node = dot, target = :out, args = (m[:in1]::NormalDistributionsFamily, m[:in2]::NormalDistributionsFamily), body = (args) -> error(DOT_OF_NORMALS))
