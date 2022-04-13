# [Meta Specification](@id user-guide-meta-specification)

Some nodes in `ReactiveMP.jl` accept optional meta structure that may be used to change or customise the inference procedure or the way node computes outbound messages. As an example `GCV` node accepts the approxximation method that will be used to approximate non-conjugate relationships between variables in this node. `GraphPPL.jl` exports `@meta` macro to specify node-specific meta and contextual information. For example:

## General syntax 

`@meta` macro accepts either regular julia function or a single `begin ... end` block. For example both are valid:

```julia

@meta function create_meta(arg1, arg2)
    ...
end

mymeta = @meta begin 
    ...
end
```

In the first case it returns a function that return meta upon calling, e.g. 

```julia
@meta function create_meta(flag)
    ...
end

mymeta = create_meta(true)
```
 
and in the second case it returns constraints directly.

```julia
mymeta = @meta begin 
    ...
end
```

## Options specification 

`@meta` macro accepts optional list of options as a first argument and specified as an array of `key = value` pairs, e.g. 

```julia
mymeta = @meta [ warn = false ] begin 
   ...
end
```

List of available options:
- `warn::Bool` - enables/disables various warnings with an incompatible model/meta specification

## Meta specification

First, lets start with an example:

```julia
meta = @meta begin 
    GCV(x, k, w) -> GCVMetadata(GaussHermiteCubature(20))
end
```

This meta specification indicates, that for every `GCV` node in the model with `x`, `k` and `w` as connected variables should use the `GCVMetadata(GaussHermiteCubature(20))` meta object.

You can have a list of as many as possible meta specification entries for different nodes:

```julia
meta = @meta begin 
    GCV(x1, k1, w1) -> GCVMetadata(GaussHermiteCubature(20))
    GCV(x2, k2, w3) -> GCVMetadata(GaussHermiteCubature(30))
    NormalMeanVariance(y, x) -> MyCustomMetaObject(arg1, arg2)
end
```

To create a model with extra constraints user may pass optional `meta` positional argument (comes either first, or after `constraints` if there are any) for the model function:

```julia
@model function my_model(arguments...)
   ...
end

constraints = @constraints begin 
    ...
end

meta = @meta begin 
    ...
end

# both are valid
model, (x, y) = my_model(meta, arguments...)
model, (x, y) = my_model(constraints, meta, arguments...)
```

Alternatively, it is possible to use [`Model`](@ref) function directly or resort to the automatic [`inference`](@ref) function that accepts `meta` keyword argument. 