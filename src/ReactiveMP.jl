module ReactiveMP

include("message.jl")
include("interface.jl")

include("variables/variables.jl")
include("variables/random.jl")
include("variables/constant.jl")
include("variables/observed.jl")
include("variables/prior.jl")

include("nodes/node.jl")
include("nodes/gaussian.jl")
include("nodes/addition.jl")
include("nodes/equality.jl")

end
