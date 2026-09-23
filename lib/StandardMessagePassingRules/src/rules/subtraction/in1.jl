# in1 = out + in2, with `convolve` for any other two distributions, as `+` towards `out`.
@define_message_update_rule(node = -, target = :in1, args = (m[:out]::NormalOrPoint, m[:in2]::NormalOrPoint), body = (args) -> sum_message(args.m[:out], args.m[:in2]))

@define_message_update_rule(node = -, target = :in1, args = (m[:out]::Distribution, m[:in2]::Distribution), body = (args) -> sum_message(args.m[:out], args.m[:in2]))

@define_message_update_rule(
    node = -, target = :in1,
    args = (m[:out]::NormalDistributionsFamily, m[:in2]::NormalDistributionsFamily),
    body = (args) -> sum_message(args.m[:out], args.m[:in2]),
)
