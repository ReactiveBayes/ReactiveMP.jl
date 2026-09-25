# in2 = out - in1.
@define_message_update_rule(node = +, target = :in2, args = (m[:out]::NormalOrPoint, m[:in1]::NormalOrPoint), logscale = 0, body = (args) -> difference_message(args.m[:out], args.m[:in1]))
