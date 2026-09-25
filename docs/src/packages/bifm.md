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

The rules keep nothing between calls. The forward rules read the message on their own edge and
recompute the quantities of the backward pass they need, keeping only working memory in their
[scratch](@ref rules-defining-scratch). They are pure, so the order they run in does not matter,
and one `BIFMSmoother` may be shared by several nodes.

!!! warning
    The free energy of a model with BIFM is not supported; asking for it raises a
    `BIFMFreeEnergyError`.
