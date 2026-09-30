# in[k] = out - the sum of the other inputs, for normal messages and known values, weighted-mean
# messages included. The message a ↦ ∫ m_out(a + b) m_others(b) db integrates to one: log scale 0,
# as for every rule of `+` and `-`.
@define_message_update_rule(
    node = +, target = (:in, k),
    args = (m[:out]::NormalOrPoint, m[:in][!k]::NormalOrPoint),
    logscale = 0,
    body = (args) -> difference_message(args.m[:out], sum_of_messages(args.m[:in])),
)
