# in2 = in1 - out.
@define_message_update_rule(node = -, target = :in2, args = (m[:out]::NormalOrPoint, m[:in1]::NormalOrPoint), logscale = 0, body = (args) -> difference_message(args.m[:in1], args.m[:out]))
