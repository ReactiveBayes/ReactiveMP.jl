# `out = in1 - in2`: the function `-` is the node. Its rules are `+`'s helpers,
# `sum_message` and `difference_message`, in the order the difference needs.
@define_factor_node(node = -, type = Deterministic, interfaces = [:out, :in1, :in2])
