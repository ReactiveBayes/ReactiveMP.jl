# [Custom Addons](@id custom-addons)

Standard message passing schemes only pass along distributions to other nodes. However, for more advanced usage, there might be a need for passing along additional information in messages and/or marginals. One can for example think of passing along the scaling of the distribution or some information that specifies how the message or marginal was computed, i.e. which messages were used for its computation and which node was preceding it.

Addons provide a solution here. Basically, addons are structures that contain extra information that are passed along the graph with messages and marginals in a tuple. Its usage and operations can differ significantly for each application, yet below gives a concise overview on how to implement them on your own.

## Example

Suppose that we wish to create an addon that counts the number of computations that preceded some message or marginal. This addon can be created by adding the file `src/addons/count.jl` and by including it in the `ReactiveMP.jl` file.

### Step 1: Creating the addon structure

Let's start by defining our new addon structure. This might seem daunting, but basically only requires us to specify the information that we would like to collect. Just make sure that it is specified as a subtype of `AbstractAddon`. In our example this becomes:
```julia
struct AddonCount{T} <: AbstractAddon
    count :: T
end
```
Aside from using this structure as an actual addon, we wish to reuse it as an option field, such that the inference algorithm knows which addons to include. In its simplest form we wish to write
```julia
inference(
    ...
    options = ( addons = ( AddonCount(), ), )
)
```
For extending the structure to an addon option field, we need to create the simplified constructor:
```julia
AddonCount() = AddonCount(nothing)
```
Step 1 is completed. You can add additional fields or functions for improved handling, such as `get_count()` or `show()` functions.

### Step 2: Computing products

The goal is to update the `AddonCount` structure when we multiply 2 messages. As a result, we need to write a function that allows us to define this behaviour. This function is called `multiply_addons` and accepts 5 arguments. In our example this becomes
```julia
function multiply_addons(left_addon::AddonCount, right_addon::AddonCount, new_dist, left_dist, right_dist)
    return AddonCount(left_addon.count + right_addon + 1)
end
```
Here we add the number of operations from the addons that are being multiplied and we add one (for the current operation). We are aware that this is likely not valid for iterative message passing schemes, but it still serves as a nice example. The `left_addon` and `right_addon` argument specify the `AddonCount` objects that are being multiplied. Corresponding to these addons, there are the distributions `left_dist` and `right_dist`, which might contain information for computing the product. The new distribution `new_dist ∝ left_dist * right_dist` is also passed along for potentially reusing the result of earlier computations.

### Step 3: Computing messages
As a final step we need to specify how the addon behaves when a new message is computed in a factor node. This step is a bit more advanced as it requires making changes in the `@rule` macro.

> Uhm this changed