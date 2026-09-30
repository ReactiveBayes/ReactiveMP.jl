# Internals

The rules do their linear and Gaussian algebra with MessagePassingRulesBase's
[math helpers](@extref MessagePassingRulesBase math-helpers), which the node packages share. This
page lists the package's own internal names.

## Products with `Uninformative`

```@docs
StandardMessagePassingRules.UninformativeProd
```
