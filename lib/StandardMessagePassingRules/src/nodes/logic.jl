# The logic nodes: deterministic functions of Bernoulli variables, true being 1.

const DOC_LOGIC_RULES = rstrip(
    """
    **Rules.** Every message is a `Bernoulli`, the probability of true, and every rule is belief
    propagation over the other interfaces' messages: a deterministic node's clusters are always
    `out` and the joint over its inputs, whatever the factorisation. The rules towards `out` have
    log scale zero; those towards an input declare the log of their normaliser. The node runs under
    [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm) and has no average
    energy of its own.
    """
)

const DOC_LOGIC_JOINT = rstrip(
    """
    The joint marginal of the inputs, `q(in1, in2)`, is a `Contingency` table over
    `{false, true}²`, rows `in1` and columns `in2`, false first.
    """
)

"""
    AND

The conjunction node, `out = in1 ∧ in2`, with interfaces `out`, `in1` and `in2`. The message towards `out` is
`Bernoulli(p₁ p₂)` for inputs `Bernoulli(p₁)` and `Bernoulli(p₂)`.

$(DOC_LOGIC_RULES)

$(DOC_LOGIC_JOINT)

# Examples

```jldoctest; setup = :(using StandardMessagePassingRules, MessagePassingRulesBase, Distributions)
julia> message = getresult(@call_message_update_rule(node = AND, target = :out, m = (in1 = Bernoulli(0.5), in2 = Bernoulli(0.2))));

julia> mean(message) ≈ 0.1
true
```

See also [`OR`](@ref), [`NOT`](@ref), [`IMPLY`](@ref).
"""
struct AND end

"""
    OR

The disjunction node, `out = in1 ∨ in2`, with interfaces `out`, `in1` and `in2`. The message
towards `out` is `Bernoulli(p₁ + p₂ - p₁ p₂)` for inputs `Bernoulli(p₁)` and `Bernoulli(p₂)`.

$(DOC_LOGIC_RULES)

$(DOC_LOGIC_JOINT)

See also [`AND`](@ref), [`NOT`](@ref), [`IMPLY`](@ref).
"""
struct OR end

"""
    NOT

The negation node, `out = ¬in`, with interfaces `out` and `in`. Each message is the other
interface's flipped: `Bernoulli(1 - p)` for `Bernoulli(p)`, both ways.

$(DOC_LOGIC_RULES) With a single input there is no joint marginal rule.

See also [`AND`](@ref), [`OR`](@ref), [`IMPLY`](@ref).
"""
struct NOT end

"""
    IMPLY

The implication node, `out = in1 → in2`, with interfaces `out`, `in1` and `in2`: false only
when `in1` is true and `in2` false. The message towards `out` is `Bernoulli(1 - p₁ + p₁ p₂)`
for inputs `Bernoulli(p₁)` and `Bernoulli(p₂)`.

$(DOC_LOGIC_RULES)

$(DOC_LOGIC_JOINT)

See also [`AND`](@ref), [`OR`](@ref), [`NOT`](@ref).
"""
struct IMPLY end

@define_factor_node(node = AND, type = Deterministic, interfaces = [:out, :in1, :in2])
@define_factor_node(node = OR, type = Deterministic, interfaces = [:out, :in1, :in2])
@define_factor_node(node = NOT, type = Deterministic, interfaces = [:out, :in])
@define_factor_node(node = IMPLY, type = Deterministic, interfaces = [:out, :in1, :in2])
