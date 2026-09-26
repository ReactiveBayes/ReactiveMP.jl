"""
    AbstractVariable

The supertype of the variables of a factor graph: [`RandomVariable`](@ref), a latent quantity;
[`DataVariable`](@ref), an observed one; and [`ConstVariable`](@ref), a constant.

A subtype implements:

- [`ReactiveMP.degree`](@ref), the number of connections to factor nodes;
- [`ReactiveMP.israndom`](@ref), [`ReactiveMP.isdata`](@ref) and [`ReactiveMP.isconst`](@ref);
- [`ReactiveMP.create_new_stream_of_inbound_messages!`](@ref), which a node's interface calls
  once per connection;
- `ReactiveMP.get_stream_of_inbound_messages(variable, index)`, the stream of the messages the
  node on connection `index` sends to the variable, and
  `ReactiveMP.get_stream_of_outbound_messages(variable, index)`, that of the messages the variable
  sends to it;
- [`ReactiveMP.get_stream_of_marginals`](@ref) and [`ReactiveMP.set_stream_of_marginals!`](@ref),
  [`ReactiveMP.get_stream_of_predictions`](@ref) and
  [`ReactiveMP.set_stream_of_predictions!`](@ref);
- `ReactiveMP.activate!(variable, options)`, when it has streams to wire.

A variable has a `label` field, which the callback events show.
"""
abstract type AbstractVariable end

Base.broadcastable(v::AbstractVariable) = Ref(v)

"""
    ReactiveMP.activate!(randomvar::RandomVariable, options::RandomVariableActivationOptions)
    ReactiveMP.activate!(datavar::DataVariable, options::DataVariableActivationOptions)
    ReactiveMP.activate!(factornode::FactorNode, options::FactorNodeActivationOptions)

Wire the lazy streams of a built graph into a live one. A graph is activated in order, once every
node is created: its random and data variables first, then the initial marginals and messages,
then its factor nodes, whose rules read the variables' streams. A constant needs no activation.
After it, each new observation propagates through the graph.

See each method: [`ReactiveMP.activate!(::RandomVariable, ::RandomVariableActivationOptions)`](@ref),
[`ReactiveMP.activate!(::DataVariable, ::DataVariableActivationOptions)`](@ref),
[`ReactiveMP.activate!(::FactorNode, ::ReactiveMP.FactorNodeActivationOptions)`](@ref).
"""
function activate! end

# Helper functions

"""
    ReactiveMP.degree(variable::AbstractVariable) -> Int

The number of factor node connections of `variable`: for a random or a data variable the number
of its inbound message streams, for a constant the number of nodes connected to it.
"""
function degree end

"""
    ReactiveMP.israndom(variable::AbstractVariable) -> Bool
    ReactiveMP.israndom(variables::AbstractArray{<:AbstractVariable}) -> Bool

Whether `variable` is a [`RandomVariable`](@ref); for an array, whether every element is. A
node interface answers for its variable.
"""
function israndom end

"""
    ReactiveMP.isdata(variable::AbstractVariable) -> Bool
    ReactiveMP.isdata(variables::AbstractArray{<:AbstractVariable}) -> Bool

Whether `variable` is a [`DataVariable`](@ref); for an array, whether every element is. A node
interface answers for its variable.
"""
function isdata end

"""
    ReactiveMP.isconst(variable::AbstractVariable) -> Bool
    ReactiveMP.isconst(variables::AbstractArray{<:AbstractVariable}) -> Bool

Whether `variable` is a [`ConstVariable`](@ref); for an array, whether every element is. A node
interface answers for its variable.
"""
function isconst end

israndom(v::AbstractArray{<:AbstractVariable}) = all(israndom, v)
isdata(v::AbstractArray{<:AbstractVariable}) = all(isdata, v)
isconst(v::AbstractArray{<:AbstractVariable}) = all(isconst, v)

"""
    ReactiveMP.create_new_stream_of_inbound_messages!(variable::AbstractVariable) -> (observable, index)

Allocate the stream of the messages a new factor node connection sends to `variable`, and return
it with the connection's index. A [`ReactiveMP.NodeInterface`](@ref) calls it when it is created,
and keeps the stream as its outbound message stream: the node's outbound message is the
variable's inbound one. The stream is a lazy [`ReactiveMP.MessageObservable`](@ref) until the node
is activated.

A random and a data variable allocate a new stream per connection, numbered from 1. A constant
counts the connection and returns its one shared stream, of the constant's message, with index
`1` every time.
"""
function create_new_stream_of_inbound_messages! end

"""
    ReactiveMP.get_stream_of_predictions(variable::AbstractVariable)

The stream of the predictions of `variable`, a [`ReactiveMP.MarginalObservable`](@ref):

- for a [`DataVariable`](@ref), the product of the messages the nodes send to it, what the
  model predicts for the observation without it; built only when the variable is activated with
  `prediction = true` (see [`DataVariableActivationOptions`](@ref)), and never emitting otherwise;
- for a [`RandomVariable`](@ref) and a [`ConstVariable`](@ref), its marginal stream.

See also [`ReactiveMP.set_stream_of_predictions!`](@ref).
"""
function get_stream_of_predictions end

"""
    ReactiveMP.set_stream_of_predictions!(variable::DataVariable, stream)

Connect the prediction stream of a data variable to `stream`, which activation does.

# Throws

- `ErrorException` for a [`RandomVariable`](@ref) and a [`ConstVariable`](@ref), whose
  predictions are their marginals.

See also [`ReactiveMP.get_stream_of_predictions`](@ref).
"""
function set_stream_of_predictions! end

"""
    ReactiveMP.get_stream_of_marginals(variable::AbstractVariable) -> MarginalObservable

The stream of the marginals of `variable`, a [`ReactiveMP.MarginalObservable`](@ref): the
posterior of a random variable, the latest observation of a data variable, the constant of a
constant. Subscribe to it to receive every update:

```julia
subscribe!(ReactiveMP.get_stream_of_marginals(x), (q) -> println(mean(q)))
```

See also [`ReactiveMP.set_initial_marginal!`](@ref), [`ReactiveMP.set_stream_of_marginals!`](@ref).
"""
function get_stream_of_marginals end

"""
    ReactiveMP.set_stream_of_marginals!(variable::AbstractVariable, stream)

Connect the marginal stream of `variable` to `stream`, which activation does.

# Throws

- `ErrorException` for a [`ConstVariable`](@ref), whose marginal is fixed.

See also [`ReactiveMP.get_stream_of_marginals`](@ref).
"""
function set_stream_of_marginals! end

"""
    ReactiveMP.set_initial_marginal!(variable::AbstractVariable, marginal)
    ReactiveMP.set_initial_marginal!(variables::AbstractArray{<:AbstractVariable}, marginals)

Seed the marginal stream of `variable` with `marginal`, a distribution, as an initial
[`Marginal`](@ref): what a rule that depends on it reads before inference computes one. For an
array, a single `PointMass` or `Distribution` seeds every variable, and a collection seeds each
variable with its element.

The initial marginal carries no log scale, even where log scales are tracked.

# Throws

- `AssertionError` when an array of variables and a collection of marginals differ in length.

See also [`ReactiveMP.set_initial_message!`](@ref).
"""
function set_initial_marginal!(variable::AbstractVariable, marginal)
    return set_initial_marginal!(get_stream_of_marginals(variable), marginal)
end

set_initial_marginal!(variables::AbstractArray{<:AbstractVariable}, marginal::PointMass) = _set_initial_marginal!(Base.HasLength(), variables, Iterators.repeated(marginal, length(variables)))
set_initial_marginal!(variables::AbstractArray{<:AbstractVariable}, marginal::Distribution) = _set_initial_marginal!(Base.HasLength(), variables, Iterators.repeated(marginal, length(variables)))
set_initial_marginal!(variables::AbstractArray{<:AbstractVariable}, marginals) = _set_initial_marginal!(Base.IteratorSize(marginals), variables, marginals)

function _set_initial_marginal!(
        ::Base.IteratorSize, variables::AbstractArray{<:AbstractVariable}, marginals
    )
    @assert length(variables) == length(marginals) "Variables $(variables) and marginals $(marginals) should have the same length"
    return foreach(zip(variables, marginals)) do (variable, marginal)
        set_initial_marginal!(variable, marginal)
    end
end

"""
    ReactiveMP.set_initial_message!(variable::RandomVariable, message)
    ReactiveMP.set_initial_message!(variables::AbstractArray{<:AbstractVariable}, messages)

Seed every message `variable` sends to its nodes with `message`, a distribution, as an initial
[`Message`](@ref): what a rule that depends on it reads before inference computes one. For an
array, a single `PointMass` or `Distribution` seeds every variable, and a collection seeds each
variable with its element.

A random variable's outbound message streams exist once it is activated, so this is called
after [`ReactiveMP.activate!`](@ref) of the variable and before that of its nodes. The initial
message's log scale is undefined.

# Throws

- `BoundsError` for a random variable not activated yet;
- `MethodError` for a [`DataVariable`](@ref), whose outbound stream holds its observations;
- `AssertionError` when an array of variables and a collection of messages differ in length.

See also [`ReactiveMP.set_initial_marginal!`](@ref).
"""
function set_initial_message!(variable::AbstractVariable, message)
    for i in 1:degree(variable)
        set_initial_message!(
            get_stream_of_outbound_messages(variable, i), message
        )
    end
    return
end

set_initial_message!(variables::AbstractArray{<:AbstractVariable}, message::PointMass) = _set_initial_message!(Base.HasLength(), variables, Iterators.repeated(message, length(variables)))
set_initial_message!(variables::AbstractArray{<:AbstractVariable}, message::Distribution) = _set_initial_message!(Base.HasLength(), variables, Iterators.repeated(message, length(variables)))
set_initial_message!(variables::AbstractArray{<:AbstractVariable}, messages) = _set_initial_message!(Base.IteratorSize(messages), variables, messages)

function _set_initial_message!(
        ::Base.IteratorSize, variables::AbstractArray{<:AbstractVariable}, messages
    )
    @assert length(variables) == length(messages) "Variables $(variables) and messages $(messages) should have the same length"
    return foreach(zip(variables, messages)) do (variable, message)
        set_initial_message!(variable, message)
    end
end
