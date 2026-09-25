@define_message_update_rule(node = DirichletCollection, target = :out, args = (m[:a]::PointMass,), logscale = 0, body = (args) -> DirichletCollection(mean(args.m[:a])))

@define_message_update_rule(node = DirichletCollection, target = :out, args = (q[:a]::PointMass,), logscale = 0, body = (args) -> DirichletCollection(mean(args.q[:a])))
