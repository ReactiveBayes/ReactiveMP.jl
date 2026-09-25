# Each bound from its message or its marginal, both point masses.
for (a, b) in ((:m, :m), (:q, :m), (:m, :q), (:q, :q))
    @eval @define_message_update_rule(
        node = Uniform, target = :out,
        args = ($a[:a]::PointMass, $b[:b]::PointMass),
        logscale = 0, body = (args) -> Uniform(mean(args.$a[:a]), mean(args.$b[:b])),
    )
end
