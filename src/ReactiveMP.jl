module ReactiveMP

include("helpers.jl")

include("distributions/gamma_ab.jl")
include("distributions/normal_mean_precision.jl")

include("message.jl")
include("messages/dummy.jl")
include("messages/delta.jl")
include("messages/normal.jl")
include("messages/normal_mean_precision.jl")
include("messages/gamma_ab.jl")

include("variable.jl")
include("variables/random.jl")
include("variables/constant.jl")
include("variables/data.jl")

include("node.jl")
include("nodes/gaussian.jl")
include("nodes/addition.jl")

include("actors/prior.jl")


# include("interface.jl")
#
# include("variables/variables.jl")
# include("variables/random.jl")
# include("variables/constant.jl")
# include("variables/observed.jl")
# include("variables/prior.jl")
#
# include("nodes/node.jl")
# include("nodes/gaussian.jl")
# include("nodes/addition.jl")
# include("nodes/equality.jl")

end
