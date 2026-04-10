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
TODO doc
"""
function degree end

"""
TODO doc
"""
function israndom end

"""
TODO doc
"""
function isdata end

"""
TODO doc
"""
function isconst end

"""
TODO doc
"""
israndom(v::AbstractArray{<:AbstractVariable}) = all(israndom, v)

"""
TODO doc
"""
isdata(v::AbstractArray{<:AbstractVariable}) = all(isdata, v)

"""
TODO doc
"""
isconst(v::AbstractArray{<:AbstractVariable}) = all(isconst, v)

"""
TODO doc
"""
function get_stream_of_predictions end

"""
TODO doc
"""
function set_stream_of_predictions! end

"""
TODO doc
"""
function get_stream_of_marginals end

"""
TODO doc
"""
function set_stream_of_marginals! end

"""
TODO doc
"""
function set_initial_marginal! end

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
TODO doc
"""
function set_initial_message! end

set_initial_message!(variables::AbstractArray{<:AbstractVariable}, message::PointMass)    = _set_initial_message!(Base.HasLength(), variables, Iterators.repeated(message, length(variables)))
set_initial_message!(variables::AbstractArray{<:AbstractVariable}, message::Distribution) = _set_initial_message!(Base.HasLength(), variables, Iterators.repeated(message, length(variables)))
set_initial_message!(variables::AbstractArray{<:AbstractVariable}, messages)              = _set_initial_message!(Base.IteratorSize(messages), variables, messages)

function _set_initial_message!(
    ::Base.IteratorSize, variables::AbstractArray{<:AbstractVariable}, messages
)
    @assert length(variables) == length(messages) "Variables $(variables) and messages $(messages) should have the same length"
    foreach(zip(variables, messages)) do (variable, message)
        setmessage!(variable, message)
    end
end
