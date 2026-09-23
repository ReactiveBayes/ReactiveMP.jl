@define_message_update_rule(node = -, target = :out, args = (m[:in1]::NormalOrPoint, m[:in2]::NormalOrPoint), body = (args) -> difference_message(args.m[:in1], args.m[:in2]))
