"""
    AbstractVariable

An abstract supertype for all variable types in the factor graph.
Concrete subtypes include:
- [`ReactiveMP.RandomVariable`](@ref)
- [`ReactiveMP.ConstVariable`](@ref)
- [`ReactiveMP.DataVariable`](@ref).
"""
abstract type AbstractVariable end

Base.broadcastable(v::AbstractVariable) = Ref(v)

# Helper functions

"""
    ReactiveMP.degree(variable)

Returns the number of factor nodes connected to `variable`, i.e. the number of message streams.
"""
function degree end

"""
    ReactiveMP.israndom(variable)
    ReactiveMP.israndom(variables::AbstractArray)

Returns `true` if `variable` is a [`ReactiveMP.RandomVariable`](@ref).
For an array, returns `true` only if all elements are random variables.
"""
function israndom end

"""
    ReactiveMP.isdata(variable)
    ReactiveMP.isdata(variables::AbstractArray)

Returns `true` if `variable` is a [`DataVariable`](@ref).
For an array, returns `true` only if all elements are data variables.
"""
function isdata end

"""
    ReactiveMP.isconst(variable)
    ReactiveMP.isconst(variables::AbstractArray)

Returns `true` if `variable` is a [`ReactiveMP.ConstVariable`](@ref).
For an array, returns `true` only if all elements are const variables.
"""
function isconst end

israndom(v::AbstractArray{<:AbstractVariable}) = all(israndom, v)
isdata(v::AbstractArray{<:AbstractVariable}) = all(isdata, v)
isconst(v::AbstractArray{<:AbstractVariable}) = all(isconst, v)

"""
    ReactiveMP.create_new_stream_of_inbound_messages!(variable)

Allocates a new per-connection [`ReactiveMP.MessageObservable`](@ref) for `variable` and registers it as an additional inbound message slot.
Returns a tuple `(observable, index)` where `observable` is the newly created stream and `index` is its position in the variable's internal `input_messages` collection.

Called once per factor node connection at graph construction time. The returned `observable` is stored as the *outbound* message stream of the corresponding [`ReactiveMP.NodeInterface`](@ref) — it is the outbound message from the node's perspective and the inbound message from the variable's perspective. All streams are unconnected (lazy) until [`ReactiveMP.activate!`](@ref) is called.

For [`ReactiveMP.ConstVariable`](@ref) the same shared observable is returned for every connection; no per-connection slot is allocated.

See also: [`ReactiveMP.MessageObservable`](@ref), [`ReactiveMP.NodeInterface`](@ref)
"""
function create_new_stream_of_inbound_messages! end

"""
    ReactiveMP.get_stream_of_predictions(variable)

Returns the prediction observable stream for `variable`.
For [`DataVariable`](@ref), the prediction is the product of all inbound messages.
See also [`ReactiveMP.set_stream_of_predictions!`](@ref).
"""
function get_stream_of_predictions end

"""
    ReactiveMP.set_stream_of_predictions!(variable, stream)

Connects `stream` as the prediction observable for `variable`.
See also [`ReactiveMP.get_stream_of_predictions`](@ref).
"""
function set_stream_of_predictions! end

"""
    ReactiveMP.get_stream_of_marginals(variable)

Returns the marginal observable stream for `variable`.
See also [`ReactiveMP.set_stream_of_marginals!`](@ref), [`ReactiveMP.set_initial_marginal!`](@ref).
"""
function get_stream_of_marginals end

"""
    ReactiveMP.set_stream_of_marginals!(variable, stream)

Connects `stream` as the marginal observable for `variable`.
See also [`ReactiveMP.get_stream_of_marginals`](@ref).
"""
function set_stream_of_marginals! end

"""
    ReactiveMP.set_initial_marginal!(variable, marginal)
    ReactiveMP.set_initial_marginal!(variables::AbstractArray, marginals)

Sets the initial marginal belief for `variable` by pushing `marginal` as an initial (non-clamped) value
into [`ReactiveMP.get_stream_of_marginals`](@ref). For arrays, applies element-wise.
See also [`ReactiveMP.set_initial_message!`](@ref).
"""
function set_initial_marginal!(variable::AbstractVariable, marginal)
    set_initial_marginal!(get_stream_of_marginals(variable), marginal)
end

set_initial_marginal!(variables::AbstractArray{<:AbstractVariable}, marginal::PointMass)    = _set_initial_marginal!(Base.HasLength(), variables, Iterators.repeated(marginal, length(variables)))
set_initial_marginal!(variables::AbstractArray{<:AbstractVariable}, marginal::Distribution) = _set_initial_marginal!(Base.HasLength(), variables, Iterators.repeated(marginal, length(variables)))
set_initial_marginal!(variables::AbstractArray{<:AbstractVariable}, marginals)              = _set_initial_marginal!(Base.IteratorSize(marginals), variables, marginals)

function _set_initial_marginal!(
    ::Base.IteratorSize, variables::AbstractArray{<:AbstractVariable}, marginals
)
    @assert length(variables) == length(marginals) "Variables $(variables) and marginals $(marginals) should have the same length"
    foreach(zip(variables, marginals)) do (variable, marginal)
        set_initial_marginal!(variable, marginal)
    end
end

"""
    ReactiveMP.set_initial_message!(variable, message)
    ReactiveMP.set_initial_message!(variables::AbstractArray, messages)

Sets the initial message for all interfaces of `variable` by pushing `message` into each outbound message stream.
For arrays, applies element-wise. See also [`ReactiveMP.set_initial_marginal!`](@ref).
"""
function set_initial_message!(variable::AbstractVariable, message)
    for i in 1:degree(variable)
        set_initial_message!(
            get_stream_of_outbound_messages(variable, i), message
        )
    end
end

set_initial_message!(variables::AbstractArray{<:AbstractVariable}, message::PointMass)    = _set_initial_message!(Base.HasLength(), variables, Iterators.repeated(message, length(variables)))
set_initial_message!(variables::AbstractArray{<:AbstractVariable}, message::Distribution) = _set_initial_message!(Base.HasLength(), variables, Iterators.repeated(message, length(variables)))
set_initial_message!(variables::AbstractArray{<:AbstractVariable}, messages)              = _set_initial_message!(Base.IteratorSize(messages), variables, messages)

function _set_initial_message!(
    ::Base.IteratorSize, variables::AbstractArray{<:AbstractVariable}, messages
)
    @assert length(variables) == length(messages) "Variables $(variables) and messages $(messages) should have the same length"
    foreach(zip(variables, messages)) do (variable, message)
        set_initial_message!(variable, message)
    end
end
