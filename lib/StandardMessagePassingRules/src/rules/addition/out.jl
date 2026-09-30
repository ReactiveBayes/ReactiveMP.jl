# The sum of the messages, left to right, for every number of inputs: normal messages and known
# values, and any other distributions through Distributions' `convolve`; normals need their own
# rule, which both of the others cover. A sum of messages integrates to one: log scale 0.
@define_message_update_rule(node = +, target = :out, args = (m[:in...]::NormalOrPoint,), logscale = 0, body = (args) -> sum_of_messages(args.m[:in]))

@define_message_update_rule(node = +, target = :out, args = (m[:in...]::Distribution,), logscale = 0, body = (args) -> sum_of_messages(args.m[:in]))

@define_message_update_rule(node = +, target = :out, args = (m[:in...]::NormalDistributionsFamily,), logscale = 0, body = (args) -> sum_of_messages(args.m[:in]))
