# [Meta Specification](@id user-guide-meta-specification)

Some nodes in `ReactiveMP.jl` accept optional meta structure that may be used to change or customise the inference procedure. As an example `GCV` node accepts the approxximation method that will be used to approximate non-conjugate relationships between variables in this node. `GraphPPL.jl` exports `@meta` macro to specify node-specific meta information. For example:

```julia
meta = @meta begin 
    GCV(x, k, w) <- GCVMetadata(GaussHermiteCubature(20))
end
```

indicates, that for every `GCV` node in the model that has `x`, `k` and `w` as connected variables the `GCVMetadata(GaussHermiteCubature(20))` meta object should be used.

`@meta` accepts function expression in the same way as `@constraints` macro, e.g:


```julia
@meta make_meta(n)
    GCV(x, k, w) <- GCVMetadata(GaussHermiteCubature(n))
end

meta = make_meta(20)
```

To create a model with extra meta options user may use optional `meta` keyword argument for the model function:

```julia
@model function my_model(arguments...)
   ...
end

meta = @meta begin 
    ...
end

model, (x, y) = model_name(arguments..., meta = meta)
```