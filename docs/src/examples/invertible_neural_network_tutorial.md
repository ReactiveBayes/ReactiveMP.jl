# [Example: Invertible Neural Network tutorial](@id examples-inn)

*Table of contents*
1. [Introduction](@ref examples-inn-introduction)
2. [Model specification](@ref examples-inn-model-specification)
3. [Model compilation](@ref examples-inn-model-compilation)
4. [Probabilistic inference](@ref examples-inn-probabilistic-inference)
5. [Parameter estimation](@ref examples-inn-parameter-estimation)

## [Introduction](@id examples-inn-introduction)

### Load required packages
Before we can start, we need to import some packages:

```@example inn
using ReactiveMP
using Rocket
using GraphPPL
using Random 
using StableRNGs

using LinearAlgebra     # only used for some matrix specifics
using Plots             # only used for visualisation
using Distributions     # only used for sampling from multivariate distributions
using Optim             # only used for parameter optimisation
```

## [Model specification](@id examples-inn-model-specification)

Specifying an invertible neural network model is easy. The general recipe looks like follows: `model = FlowModel(input_dim, (layer1(options), layer2(options), ...))`. Here the first argument corresponds to the input dimension of the model and the second argument is a tuple of layers. An example model can be defined as

```@example inn
model = FlowModel(2,
    (
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(PlanarFlow(); permute=false)
    )
)
nothing #hide
```

Alternatively, the `input_dim` can also be passed as an `InputLayer` layer as 

```@exampel inn
model = FlowModel(
    (
        InputLayer(2),
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(PlanarFlow(); permute=false)
    )
)
nothing #hide
```

In the above `AdditiveCouplingLayer` layers the input ``{\bf{x}} = [x_1, x_2, \ldots, x_N]`` is partitioned into chunks of unit length. These partitions are additively coupled to an output ``{\bf{y}} = [y_1, y_2, \ldots, y_N]`` as 

```math
\begin{align*}
    y_1 &= x_1 \\
    y_2 &= x_2 + f_1(x_1) \\
    \vdots \\
    y_N &= x_N + f_{N-1}(x_{N-1})
\end{align*}
```math

Importantly, this structure can easily be converted as 

```math
\begin{align*}
    x_1 &= y_1 \\
    x_2 &= y_2 - f_1(x_1) \\
    \vdots \\
    x_N &= y_N - f_{N-1}(x_{N-1})
\end{align*}
```

$f_n$ is an arbitrarily complex function, here chosen to be a `PlanarFlow`, but this can be interchanged for any function or neural network. The `permute` keyword argument (which defaults to `true`) specifies whether the output of this layer should be randomly permuted or shuffled. This makes sure that the first element is also transformed in consecutive layers.

A permutation layer can also be added by itself as a `PermutationLayer` layer with a custom permutation matrix if desired.

```@example inn
model = FlowModel(
    (
        InputLayer(2),
        AdditiveCouplingLayer(PlanarFlow(); permute=false),
        PermutationLayer(PermutationMatrix(2)),
        AdditiveCouplingLayer(PlanarFlow(); permute=false)
    )
)
nothing #hide
```

## [Model compilation](@id examples-inn-model-compilation)
In the current models, the layers are setup to work with the passed input dimension. This means that the function $f_n$ is repeated `input_dim-1` times for each of the partitions. Furthermore the permutation layers are set up with proper permutation matrices. If we print the model we get

```@example inn
model
```

The text below describes the terms above. Please note the distinction in typing and elements, i.e. `FlowModel{types}(elements)`:
- `FlowModel` - specifies that we are dealing with an invertible neural network model.
- `3` - Number of layers.
- `Tuple{AdditiveCouplingLayerEmpty{...},PermutationLayer{Int64},AdditiveCouplingLayerEmpty{...}}` - tuple of layer types.
- `Tuple{ReactiveMP.PlanarFlowEmpty{1},ReactiveMP.PlanarFlowEmpty{1}}` - tuple of functions $f_n$.
- `PermutationLayer{Int64}(2, [0 1; 1 0])` - permutation layer with input dimension 2 and permutation matrix `[0 1; 1 0]`.

From inspection we can see that the `AdditiveCouplingLayerEmpty` and `PlanarFlowEmpty` objects are different than before. They are initialized for the correct dimension, but they do not have any parameters registered to them. This is by design to allow for separating the model specification from potential optimization procedures. Before we perform inference in this model, the parameters should be initialized. We can randomly initialize the parameters as

```@example inn
compiled_model = compile(model)
```

## [Probabilistic inference](@id examples-inn-probabilistic-inference)

We can perform inference in our compiled model through standard usage of ReactiveMP. Let's first generate some random 2D data which has been sampled from a standard normal distribution and is consecutively passed through an invertible neural network. Using the `forward(model, data)` function we can propagate data in the forward direction.

```@example inn
function generate_data(nr_samples::Int64, model::CompiledFlowModel; seed = 123)

    rng = StableRNG(seed)
    
    # specify latent sampling distribution
    dist = MvNormal([1.5, 0.5], I)

    # sample from the latent distribution
    x = rand(rng, dist, nr_samples)

    # transform data
    y = zeros(Float64, size(x))
    for k = 1:nr_samples
        y[:,k] .= ReactiveMP.forward(model, x[:,k])
    end

    # return data
    return y, x

end;
```

```@example inn
# generate data
y, x = generate_data(1000, compiled_model)

# plot generated data
p1 = scatter(x[1,:], x[2,:], alpha=0.3, title="Original data", size=(800,400))
p2 = scatter(y[1,:], y[2,:], alpha=0.3, title="Transformed data", size=(800,400))
plot(p1, p2, legend = false)
```

The probabilistic model for doing inference can be described as 

```@example inn
@model function invertible_neural_network(nr_samples::Int64)
    
    # initialize variables
    z_μ   = randomvar()
    z_Λ   = randomvar()
    x     = randomvar(nr_samples)
    y_lat = randomvar(nr_samples)
    y     = datavar(Vector{Float64}, nr_samples)

    # specify prior
    z_μ ~ MvNormalMeanCovariance(zeros(2), huge*diagm(ones(2)))
    z_Λ ~ Wishart(2.0, tiny*diagm(ones(2)))

    # specify observations
    for k = 1:nr_samples

        # specify latent state
        x[k] ~ MvNormalMeanPrecision(z_μ, z_Λ)

        # specify transformed latent value
        y_lat[k] ~ Flow(x[k])

        # specify observations
        y[k] ~ MvNormalMeanCovariance(y_lat[k], tiny*diagm(ones(2)))

    end

    # return variables
    return z_μ, z_Λ, x, y_lat, y

end;
```

Here the inverse neural network model is passed inside a meta data object of the flow node.
Inference then resorts to

```@example inn
observations = [y[:,k] for k=1:size(y,2)]

fmodel         = Model(invertible_neural_network, length(observations))
data          = (y = observations, )
initmarginals = (z_μ = MvNormalMeanCovariance(zeros(2), huge*diagm(ones(2))), z_Λ = Wishart(2.0, tiny*diagm(ones(2))))
returnvars    = (z_μ = KeepLast(), z_Λ = KeepLast(), x = KeepLast(), y_lat = KeepLast())

constraints = @constraints begin
    q(z_μ, x, z_Λ) = q(z_μ)q(z_Λ)q(x)
end

@meta function fmeta(model)
    compiled_model = compile(model, randn(StableRNG(321), nr_params(model)))
    Flow(y_lat, x) -> FlowMeta(compiled_model) # defaults to FlowMeta(compiled_model; approximation=Linearization()). 
                                               # other approximation methods can be e.g. FlowMeta(compiled_model; approximation=Unscented(input_dim))
end

# First execution is slow due to Julia's initial compilation 
result = inference(
    model = fmodel, 
    data  = data,
    constraints   = constraints,
    meta          = fmeta(model),
    initmarginals = initmarginals,
    returnvars    = returnvars,
    free_energy   = true,
    iterations    = 10, 
    showprogress  = false
)
```

The following line of code then executes the inference algorithm.

```@example inn
fe_flow = result.free_energy
zμ_flow = result.posteriors[:z_μ]
zΛ_flow = result.posteriors[:z_Λ]
x_flow  = result.posteriors[:x]
y_flow  = result.posteriors[:y_lat]
nothing #hide
```

As we can see, the variational free energy decreases inside of our model.

```@example inn
plot(1:10, fe_flow/size(y,2), xlabel="iteration", ylabel="normalized variational free energy [nats/sample]", legend=false)
```

If we plot a random noisy observation and its approximated transformed uncertainty we obtain:

```@example inn
# pick a random observation
id = rand(StableRNG(321), 1:size(y,2))
rand_observation = MvNormal(y[:,id], 5e-1*diagm(ones(2)))
warped_observation = MvNormal(ReactiveMP.backward(compiled_model, y[:,id]), ReactiveMP.inv_jacobian(compiled_model, y[:,id])*5e-1*diagm(ones(2))*ReactiveMP.inv_jacobian(compiled_model, y[:,id])');

p1 = scatter(x[1,:], x[2,:], alpha=0.1, title="Latent distribution", size=(1200,500), label="generated data")
contour!(-5:0.1:5, -5:0.1:5, (x, y) -> pdf(MvNormal([1.5, 0.5], I), [x, y]), c=:viridis, colorbar=false, linewidth=2)
scatter!([mean(zμ_flow)[1]], [mean(zμ_flow)[2]], color="red", markershape=:x, markersize=5, label="inferred mean")
contour!(-5:0.01:5, -5:0.01:5, (x, y) -> pdf(warped_observation, [x, y]), colors="red", levels=1, linewidth=2, colorbar=false)
scatter!([mean(warped_observation)[1]], [mean(warped_observation)[2]], color="red", label="transformed noisy observation")
p2 = scatter(y[1,:], y[2,:], alpha=0.1, label="generated data")
scatter!([ReactiveMP.forward(compiled_model, mean(zμ_flow))[1]], [ReactiveMP.forward(compiled_model, mean(zμ_flow))[2]], color="red", marker=:x, label="inferred mean")
contour!(-10:0.1:10, -10:0.1:10, (x, y) -> pdf(MvNormal([1.5, 0.5], I), ReactiveMP.backward(compiled_model, [x, y])), c=:viridis, colorbar=false, linewidth=2)
contour!(-10:0.1:10, -10:0.1:10, (x, y) -> pdf(rand_observation, [x, y]), colors="red", levels=1, linewidth=2, label="random noisy observation", colorba=false)
scatter!([mean(rand_observation)[1]], [mean(rand_observation)[2]], color="red", label="random noisy observation")
plot(p1, p2, legend = true)
```

## [Parameter estimation](@id examples-inn-parameter-estimation)

The invertible neural network model is often used to learn unknown probabilistic mappings. Here we will demonstrate it as follows for a binary classification task with the following data:

```@example inn
function generate_data(nr_samples::Int64; seed = 123)
    
    rng = StableRNG(seed)

    # sample weights
    w = rand(rng, nr_samples, 2)

    # sample appraisal
    y = zeros(Float64, nr_samples)
    for k = 1:nr_samples
        y[k] = 1.0*(w[k,1] > 0.5)*(w[k,2] < 0.5)
    end

    # return data
    return y, w

end;
```

```@example inn
data_y, data_x = generate_data(50);
scatter(data_x[:,1], data_x[:,2], marker_z=data_y, xlabel="w1", ylabel="w2", colorbar=false, legend=false)
```

We will then specify a possible model as

```@example inn
# specify flow model
model = FlowModel(2,
    (
        AdditiveCouplingLayer(PlanarFlow()), # defaults to AdditiveCouplingLayer(PlanarFlow(); permute=true)
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(PlanarFlow(); permute=false)
    )
);
```

The corresponding probabilistic model for the binary classification task can be created as

```@example inn
@model [ default_factorisation = FullFactorisation() ] function invertible_neural_network_classifier(nr_samples::Int64)
    
    # initialize variables
    x_lat  = randomvar(nr_samples)
    y_lat1 = randomvar(nr_samples)
    y_lat2 = randomvar(nr_samples)
    y      = datavar(Float64, nr_samples)
    x      = datavar(Vector{Float64}, nr_samples)

    # specify observations
    for k = 1:nr_samples

        # specify latent state
        x_lat[k] ~ MvNormalMeanPrecision(x[k], 1e3*diagm(ones(2)))

        # specify transformed latent value
        y_lat1[k] ~ Flow(x_lat[k])
        y_lat2[k] ~ dot(y_lat1[k], [1, 1])

        # specify observations
        y[k] ~ Probit(y_lat2[k]) # default: where { pipeline = RequireMessage(in = NormalMeanPrecision(0, 1.0)) }

    end

    # return variables
    return x_lat, x, y_lat1, y_lat2, y

end
```

```@example inn 
fcmodel       = Model(invertible_neural_network_classifier, length(data_y))
data          = (y = data_y, x = [data_x[k,:] for k=1:size(data_x,1)], )

@meta function fmeta(model, params)
    compiled_model = compile(model, params)
    Flow(y_lat1, x_lat) -> FlowMeta(compiled_model)
end
```

Here we see that the compilation occurs inside of our probabilistic model. As a result we can pass parameters (and a model) to this function which we wish to opmize for some criterium, such as the variational free energy. Inference can be described as

For the optimization procedure, we will simplify our inference loop, such that it only accepts parameters as an argument (which is wishes to optimize) and outputs a performance metric.

```@example inn
function f(params)
    Random.seed!(42) # Flow uses random permutation matrices, which is not good for the optimisation procedure
    result = inference(
        model                   = fcmodel, 
        data                    = data,
        meta                    = fmeta(model, params),
        free_energy             = true,
        free_energy_diagnostics = nothing, # Free Energy can be set to NaN due to optimization procedure
        iterations              = 10, 
        showprogress            = false
    );
    
    result.free_energy[end]
end;
```

Optimization can be performed using the `Optim` package. Alternatively, other (custom) optimizers can be implemented, such as:

```julia
res = optimize(f, randn(StableRNG(42), nr_params(model)), GradientDescent(), Optim.Options(f_tol = 1e-3, store_trace = true, show_trace = true, show_every = 100), autodiff=:forward)
``` 

- uses finitediff and is slower/less accurate.

*or*

```julia
# create gradient function
g = (x) -> ForwardDiff.gradient(f, x);

# specify initial params
params = randn(nr_params(model))

# create custom optimizer (here Adam)
optimizer = Adam(params; λ=1e-1)

# allocate space for gradient
∇ = zeros(nr_params(model))

# perform optimization
for it = 1:10000

    # backward pass
    ∇ .= ForwardDiff.gradient(f, optimizer.x)

    # gradient update
    ReactiveMP.update!(optimizer, ∇)

end

```

```@example inn
res = optimize(f, randn(StableRNG(42), nr_params(model)), GradientDescent(), Optim.Options(store_trace = true, show_trace = true, show_every = 50), autodiff=:forward)
nothing #hide
```

optimization results are then given as

```@example inn
params = Optim.minimizer(res)
inferred_model = compile(model, params)
trans_data_x_1 = hcat(map((x) -> ReactiveMP.forward(inferred_model, x), [data_x[k,:] for k=1:size(data_x,1)])...)'
trans_data_x_2 = map((x) -> dot([1, 1], x), [trans_data_x_1[k,:] for k=1:size(data_x,1)])
trans_data_x_2_split = [trans_data_x_2[data_y .== 1.0], trans_data_x_2[data_y .== 0.0]]
p1 = scatter(data_x[:,1], data_x[:,2], marker_z = data_y, size=(1200,400), c=:viridis, colorbar=false, title="original data")
p2 = scatter(trans_data_x_1[:,1], trans_data_x_1[:,2], marker_z = data_y, c=:viridis, size=(1200,400), colorbar=false, title="|> warp")
p3 = histogram(trans_data_x_2_split; stacked=true, bins=50, size=(1200,400), title="|> dot")
plot(p1, p2, p3, layout=(1,3), legend=false)
```

```@example inn
using StatsFuns: normcdf
p1 = scatter(data_x[:,1], data_x[:,2], marker_z = data_y, title="original labels", xlabel="weight 1", ylabel="weight 2", size=(1200,400), c=:viridis)
p2 = scatter(data_x[:,1], data_x[:,2], marker_z = normcdf.(trans_data_x_2), title="predicted labels", xlabel="weight 1", ylabel="weight 2", size=(1200,400), c=:viridis)
p3 = contour(0:0.01:1, 0:0.01:1, (x, y) -> normcdf(dot([1,1], ReactiveMP.forward(inferred_model, [x,y]))), title="Classification map", xlabel="weight 1", ylabel="weight 2", size=(1200,400), c=:viridis)
plot(p1, p2, p3, layout=(1,3), legend=false)
```
