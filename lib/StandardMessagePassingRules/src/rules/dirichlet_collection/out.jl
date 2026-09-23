@define_message_update_rule(node = DirichletCollection, target = :out, args = (m[:a]::PointMass,), body = (args) -> DirichletCollection(mean(args.m[:a])))

@define_message_update_rule(node = DirichletCollection, target = :out, args = (q[:a]::PointMass,), body = (args) -> DirichletCollection(mean(args.q[:a])))
