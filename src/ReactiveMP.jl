
module ReactiveMP

# List global dependencies here
using TinyHugeNumbers, MatrixCorrectionTools, FastCholesky, LinearAlgebra
using BayesBase, ExponentialFamily
using UUIDs

import MatrixCorrectionTools: AbstractCorrectionStrategy, correction!

# Reexport `tiny` and `huge` from the `TinyHugeNumbers`
export tiny, huge

include("fixes.jl")
include("helpers/macrohelpers.jl")
include("helpers/helpers.jl")

include("helpers/algebra/companion_matrix.jl")
include("helpers/algebra/common.jl")
include("helpers/algebra/permutation_matrix.jl")
include("helpers/algebra/standard_basis_vector.jl")

include("constraints/form.jl")

include("callbacks.jl")
include("postprocessors.jl")
include("variable.jl")
include("annotations.jl")
include("annotations/logscale.jl")
include("annotations/input_arguments.jl")
include("message.jl")
include("marginal.jl")

"""
    to_marginal(any)

Transforms an input to a proper marginal distribution.
Called inside `as_marginal`. Some nodes do not use `Distributions.jl`, but instead implement their own equivalents for messages for better efficiency.
Effectively `to_marginal` is needed to convert internal effective implementation to a user-friendly equivalent (e.g. from `Distributions.jl`).
By default does nothing and returns its input, but some nodes may override this behaviour (see for example `Wishart` and `InverseWishart`).

Note: This function is a part of the private API and is not intended to be used outside of the ReactiveMP package.
"""
to_marginal(any) = any

as_marginal(message::Message)  = Marginal(to_marginal(getdata(message)), is_clamped(message), is_initial(message), getannotations(message))
as_message(marginal::Marginal) = Message(getdata(marginal), is_clamped(marginal), is_initial(marginal), getannotations(marginal))

getdata(::Nothing)                 = nothing
getdata(collection::Tuple)         = map(getdata, collection)
getdata(collection::AbstractArray) = map(getdata, collection)

# TupleTools.prod is a more efficient version of Base.all for Tuple here
is_clamped(tuple::Tuple) = TupleTools.prod(map(is_clamped, tuple))
is_initial(tuple::Tuple) = TupleTools.prod(map(is_initial, tuple))

include("approximations/approximations.jl")
include("approximations/shared.jl")
include("approximations/gausshermite.jl")
include("approximations/gausslaguerre.jl")
include("approximations/sphericalradial.jl")
include("approximations/laplace.jl")
include("approximations/importance.jl")
include("approximations/optimizers.jl")
include("approximations/rts.jl")
include("approximations/linearization.jl")
include("approximations/unscented.jl")
include("approximations/cvi.jl")
include("approximations/cvi_projection.jl")

# Predefined postprocessors
include("postprocessors/scheduled.jl")

# Equality node is a special case and needs to be included before random variable implementation
include("nodes/equality.jl")

include("variables/random.jl")
include("variables/constant.jl")
include("variables/data.jl")

include("nodes/nodes.jl")
include("rule.jl")

include("score/score.jl")
include("score/variable.jl")
include("score/node.jl")

include("nodes/predefined.jl")
include("rules/predefined.jl")
include("rules/fallbacks.jl")

function __init__()
    Base.Experimental.register_error_hint(
        MethodError
    ) do io, exc, argtypes, kwargs
        if exc.f == ReactiveMP.factornode &&
            length(argtypes) >= 2 &&
            argtypes[1] == ReactiveMP.UndefinedNodeFunctionalForm
            errmsg = """
            `$(argtypes[2])` has been used but the `ReactiveMP` backend does not support `$(argtypes[2])` as a factor node.

            Please refer to the [factor nodes](https://reactivebayes.github.io/ReactiveMP.jl/stable/lib/nodes/) section of the documentation for more details.
            """
            println(io, errmsg)
        end
        if exc.f === ReactiveMP.handle_event && length(argtypes) >= 2
            event_type = argtypes[2]
            event_hint = if event_type <: ReactiveMP.Event
                "Event{$(repr(ReactiveMP.event_name(event_type)))}"
            else
                string(event_type)
            end
            errmsg = """

            `ReactiveMP.handle_event` was called with a callback handler of type `$(argtypes[1])` for event `$(event_type)`, but no matching method was found. This can happen if:

            1. You implemented a custom callback handler but forgot to define `handle_event` for this specific event type.
               Make sure your handler has a method like:
                 ReactiveMP.handle_event(::$(argtypes[1]), event::$(event_hint)) = ...

            2. You meant to pass a `NamedTuple` as the callbacks handler but forgot the trailing comma.
               In Julia, `(key = value)` is parsed as a plain assignment, not a NamedTuple.
               Use `(key = value,)` (with a trailing comma) instead.
            """
            println(io, errmsg)
        end
    end
end

end
