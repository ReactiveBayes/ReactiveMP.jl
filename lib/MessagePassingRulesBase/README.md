# MessagePassingRulesBase

The rule system of the ReactiveMP ecosystem: the macros that declare factor nodes, their message
update rules, marginal rules, average energies and dependencies, and the lookup an engine uses to
find and run a rule for a node, a target and the types of its inputs. Rules are ordinary Julia
functions, found through dispatch whichever loaded package defines them, and callable and
testable without an engine. It also holds the math helpers the rule packages share, linear and
Gaussian algebra such as `add_outer` and `gaussian_second_moment`, public and documented. Every
rule package under `lib/` builds on it, and ReactiveMP runs its rules.

```julia
using MessagePassingRulesBase

struct Shift end   # out = in + 1

@define_factor_node(node = Shift, type = Deterministic, interfaces = [:out, :in])

@define_message_update_rule(
    node = Shift, target = :out, args = (m[:in]::Real,), logscale = 0,
    body = (args) -> args.m[:in] + 1,
)

result = @call_message_update_rule(node = Shift, target = :out, m = (in = 1.0,))
getresult(result), getlogscale(result)   # (2.0, 0)
```

- Documentation: `make docs-base` from the repository root builds it into `docs/build`; it will
  be published at <https://reactivebayes.github.io/MessagePassingRulesBase.jl/dev/>.
- Tests: `make test-base`; `TEST_ALL=true make test-base` includes the items tagged `:slow`.
- Depends on BayesBase, MacroTools, Compat, FastCholesky, IrrationalConstants and LinearAlgebra:
  no distribution package and no engine.
  Julia 1.10 or later. MIT licence.
