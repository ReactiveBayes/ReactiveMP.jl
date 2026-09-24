# [BIFM](@id packages-bifm)

```@docs
BIFMMessagePassingRules
```

`BIFM` is a whole time slice of a linear state-space model, `znext = A zprev + B in` and
`out = C znext`, for backward-information-filter forward-marginal smoothing. The backward pass
sends information-form messages towards the earlier states; `BIFMHelper`, at the start of the
chain, turns it into the forward pass, whose messages are the marginals themselves
(`TerminalProdArgument`s). On a chain the result is the RTS smoother's, which the engine's tests
check.

```@docs
BIFM
BIFMSmoother
BIFMHelper
BIFMMessagePassingRules.BIFMFreeEnergyError
```

!!! note
    v6's `BIFMMeta` was a mutable cache: the rule towards `zprev` stored intermediate quantities
    that the other rules read back, so results depended on the order the rules ran in, and one
    meta shared by two nodes corrupted both. The port's forward rules read the message on their
    own edge and recompute those quantities, keeping only working memory in their
    [scratch](@ref rules-defining-scratch). They are pure and independent of order.

!!! warning
    The free energy of a model with BIFM is not supported, as in v6, where it failed with an
    infinite node bound; asking for it raises a `BIFMFreeEnergyError`.
