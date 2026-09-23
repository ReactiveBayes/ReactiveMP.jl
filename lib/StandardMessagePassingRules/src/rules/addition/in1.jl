# in1 = out - in2. v6's specialisation for two BLAS-typed MvNormalWeightedMeanPrecision messages
# took μ_in2 - μ_out (ReactiveMP.jl#677); it is not ported, the generic rule covers them.
@define_message_update_rule(node = +, target = :in1, args = (m[:out]::NormalOrPoint, m[:in2]::NormalOrPoint), body = (args) -> difference_message(args.m[:out], args.m[:in2]))
