# The logic nodes: deterministic functions of Bernoulli variables, true being 1. Their rules
# are belief propagation under the default scheme, and the joint over their inputs a
# `Contingency` table, rows `in1` and columns `in2`, false first.

"""
    AND

Conjunction, `out = in1 ∧ in2`.
"""
struct AND end

"""
    OR

Disjunction, `out = in1 ∨ in2`.
"""
struct OR end

"""
    NOT

Negation, `out = ¬in`.
"""
struct NOT end

"""
    IMPLY

Implication, `out = in1 → in2`: false only when `in1` is true and `in2` false.
"""
struct IMPLY end

@define_factor_node(node = AND, type = Deterministic, interfaces = [:out, :in1, :in2])
@define_factor_node(node = OR, type = Deterministic, interfaces = [:out, :in1, :in2])
@define_factor_node(node = NOT, type = Deterministic, interfaces = [:out, :in])
@define_factor_node(node = IMPLY, type = Deterministic, interfaces = [:out, :in1, :in2])
