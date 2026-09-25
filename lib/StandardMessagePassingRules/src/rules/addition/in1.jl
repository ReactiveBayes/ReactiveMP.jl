# in1 = out - in2, for every pair of inputs, weighted-mean messages included. The message
# a ↦ ∫ m_out(a + b) m_in2(b) db integrates to one: log scale 0, as for every rule of `+` and `-`.
@define_message_update_rule(node = +, target = :in1, args = (m[:out]::NormalOrPoint, m[:in2]::NormalOrPoint), logscale = 0, body = (args) -> difference_message(args.m[:out], args.m[:in2]))
