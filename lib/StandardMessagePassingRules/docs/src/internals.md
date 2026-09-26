# Internals

Helpers the rules share, and that the node packages call qualified, as
`StandardMessagePassingRules.mul_trace`. They are not part of the public API. The in-place ones
overwrite a dense `Array` and leave any other argument untouched, so a rule may pass them a view
or a number.

```@docs
StandardMessagePassingRules.v_a_vT
StandardMessagePassingRules.negate_inplace!
StandardMessagePassingRules.mul_inplace!
StandardMessagePassingRules.rank1update
StandardMessagePassingRules.mul_trace
```

## Products with `Uninformative`

```@docs
StandardMessagePassingRules.UninformativeProd
```
