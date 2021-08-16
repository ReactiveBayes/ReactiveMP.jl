
# [Helper utilities](@id lib-helpers)

## [OneDivNVector](@id lib-one-div-n-vector)

Helper utilities implement [`OneDivNVector`](@ref) structure that is allocation free equivalent of `fill(1 / N, N)` collection. Mostly used in [`SampleList`](@ref) implementation.

```@meta
DocTestSetup = quote
    using ReactiveMP
    import ReactiveMP: OneDivNVector
end
```

```@docs
ReactiveMP.OneDivNVector
```

```@meta
DocTestSetup = nothing
```