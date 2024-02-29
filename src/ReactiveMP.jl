
module ReactiveMP

# List global dependencies here
using TinyHugeNumbers, MatrixCorrectionTools, FastCholesky, LinearAlgebra
using BayesBase, ExponentialFamily

import MatrixCorrectionTools: AbstractCorrectionStrategy, correction!

# Reexport `tiny` and `huge` from the `TinyHugeNumbers`
export tiny, huge

include("fixes.jl")
include("helpers/macrohelpers.jl")
include("helpers/helpers.jl")
include("helpers/manyof.jl")

include("helpers/algebra/companion_matrix.jl")
include("helpers/algebra/common.jl")
include("helpers/algebra/permutation_matrix.jl")
include("helpers/algebra/standard_basis_vector.jl")

include("constraints/form.jl")

include("message.jl")
include("marginal.jl")
include("addons.jl")

include("addons/debug.jl")
include("addons/logscale.jl")
include("addons/memory.jl")

"""
    to_marginal(any)

Transforms an input to a proper marginal distribution.
Called inside `as_marginal`. Some nodes do not use `Distributions.jl`, but instead implement their own equivalents for messages for better efficiency.
Effectively `to_marginal` is needed to convert internal effective implementation to a user-friendly equivalent (e.g. from `Distributions.jl`).
By default does nothing and returns its input, but some nodes may override this behaviour (see for example `Wishart` and `InverseWishart`).

Note: This function is a part of the private API and is not intended to be used outside of the ReactiveMP package.
"""
to_marginal(any) = any

as_marginal(message::Message)  = Marginal(to_marginal(getdata(message)), is_clamped(message), is_initial(message), getaddons(message))
as_message(marginal::Marginal) = Message(getdata(marginal), is_clamped(marginal), is_initial(marginal), getaddons(marginal))

getdata(::Nothing)                 = nothing
getdata(collection::Tuple)         = map(getdata, collection)
getdata(collection::AbstractArray) = map(getdata, collection)

getlogscale(message::Message)      = getlogscale(getaddons(message))
getlogscale(marginal::Marginal)    = getlogscale(getaddons(marginal))
getmemoryaddon(message::Message)   = getmemoryaddon(getaddons(message))
getmemoryaddon(marginal::Marginal) = getmemoryaddon(getaddons(marginal))
getmemory(message::Message)        = getmemory(getaddons(message))
getmemory(marginal::Marginal)      = getmemory(getaddons(marginal))

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

# Equality node is a special case and needs to be included before random variable implementation
include("nodes/equality.jl")

include("variables/variable.jl")
include("variables/random.jl")
include("variables/constant.jl")
include("variables/data.jl")

include("pipeline/pipeline.jl")
include("pipeline/async.jl")
include("pipeline/discontinue.jl")
include("pipeline/logger.jl")

include("nodes/nodes.jl")

include("rule.jl")

include("score/score.jl")
include("score/variable.jl")
include("score/node.jl")

include("nodes/predefined.jl")
include("rules/predefined.jl")


# This symbol is only defined on Julia versions that support extensions
@static if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()

    # A backward-compatible solution for older versions of Julia
    # For Julia > 1.9 this will be loaded automatically without need in `Requires.jl`
    @static if !isdefined(Base, :get_extension)
        @require Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2" include("../ext/ReactiveMPOptimisersExt/ReactiveMPOptimisersExt.jl")
        @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include("../ext/ReactiveMPZygoteExt/ReactiveMPZygoteExt.jl")
    end
end

end
