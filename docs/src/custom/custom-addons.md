# [Custom Addons](@id custom-addons)

Standard message passing schemes only pass along distributions to other nodes. However, for more advanced usage, there might be a need for passing along additional information in messages and/or marginals. One can for example think of passing along the scaling of the distribution or some information that specifies how the message or marginal was computed, i.e. which messages were used for its computation and which node was preceding it. Another use cases is saving extra debugging information inside messages themselves, e.g. what arguments have been used to compute a message.

Addons provide a solution here. Basically, addons are structures that contain extra information that are passed along the graph with messages and marginals in a tuple. These addons can be extracted using the `getaddons(message/marginal)` function. Its usage and operations can differ significantly for each application, yet below gives a concise overview on how to implement them on your own.

## Example

Suppose that we wish to create an addon that counts the number of computations that preceded some message or marginal. This addon can be created by adding the file `src/addons/count.jl` and by including it in the `ReactiveMP.jl` file.

### Step 1: Creating the addon structure

Let's start by defining our new addon structure. This might seem daunting, but basically only requires us to specify the information that we would like to collect. Just make sure that it is specified as a subtype of `AbstractAddon`. In our example this becomes:

```julia
struct AddonCount{T} <: AbstractAddon
    count :: T
end
```

You can add additional fields or functions for improved handling, such as `get_count()` or `show()` functions.

### Step 2: Compute addon value after computing a message

As a second step we need to specify how the addon behaves when a new message is computed in a factor node. 
For this purpose we need to implement a specialized version of the `message_mapping_addon()` function. This function accepts the mapping variables of the factor node and updates the addons by extending the tuple.

In our example we could write
```julia
# This specification assumes that the default value for addon is `AddonCount(nothing)`
function message_mapping_addon(::AddonCount{Nothing}, mapping, messages, marginals, result, addons)

    # get number of operations of messages
    message_count = 0
    for message in messages
        message_count += getcount(message)
    end

    # get number of operations of marginals
    marginal_count = 0
    for marginal in marginals
        marginal_count += getcount(marginal)
    end

    # extend addons with AddonCount() structure
    return AddonCount(message_count + marginal_count + 1)
end

```
### Step 3: Computing products

The goal is to update the `AddonCount` structure when we multiply 2 messages. As a result, we need to write a function that allows us to define this behaviour. This function is called `multiply_addons` and accepts 5 arguments. In our example this becomes

```julia
function multiply_addons(left_addon::AddonCount, right_addon::AddonCount, new_dist, left_dist, right_dist)
    return AddonCount(left_addon.count + right_addon.count + 1)
end
```

here we add the number of operations from the addons that are being multiplied and we add one (for the current operation). we are aware that this is likely not valid for iterative message passing schemes, but it still serves as a nice example. the `left_addon` and `right_addon` argument specify the `addoncount` objects that are being multiplied. corresponding to these addons, there are the distributions `left_dist` and `right_dist`, which might contain information for computing the product. the new distribution `new_dist ∝ left_dist * right_dist` is also passed along for potentially reusing the result of earlier computations.

### More information

For more advanced information check the implementation of the log-scale or memory addons.
