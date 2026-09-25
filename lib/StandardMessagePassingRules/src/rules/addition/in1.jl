# in1 = out - in2, for every pair of inputs, weighted-mean messages included.
@define_message_update_rule(node = +, target = :in1, args = (m[:out]::NormalOrPoint, m[:in2]::NormalOrPoint), body = (args) -> difference_message(args.m[:out], args.m[:in2]))
