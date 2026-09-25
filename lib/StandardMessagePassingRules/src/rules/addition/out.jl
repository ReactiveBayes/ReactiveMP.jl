@define_message_update_rule(node = +, target = :out, args = (m[:in1]::NormalOrPoint, m[:in2]::NormalOrPoint), logscale = 0, body = (args) -> sum_message(args.m[:in1], args.m[:in2]))

# Any other two distributions, through Distributions' `convolve`; two normals need their own
# rule, which both of the others cover.
@define_message_update_rule(node = +, target = :out, args = (m[:in1]::Distribution, m[:in2]::Distribution), logscale = 0, body = (args) -> sum_message(args.m[:in1], args.m[:in2]))

@define_message_update_rule(
    node = +, target = :out,
    args = (m[:in1]::NormalDistributionsFamily, m[:in2]::NormalDistributionsFamily),
    logscale = 0,
    body = (args) -> sum_message(args.m[:in1], args.m[:in2]),
)
