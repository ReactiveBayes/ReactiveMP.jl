
struct Belief{D} <: AbstractBelief{D}
    data :: D
end

data(belief::Belief) = belief.data
