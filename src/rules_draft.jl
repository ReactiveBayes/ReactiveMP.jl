
# Draft for m_ja message computation rule
# First argument is a Node functional form, can be ::Normal or for example ::typeof(sin) in nonlinear case
# Subject to discuss
function rule(::Node{Normal}, ...) end
function rule(::Normal, ...) end

# Second argument essentially means edge index in m_ja, probably should be readable, like 'mean' or 'presicion'
# names should be predefined for every node (not a big deal, right?). yea, it is possible to dispatch on symbol
function rule(::Normal, ::Val{:out}, ...) end
function rule(::Normal, ::Val{:mean}, ...) end
# Ismail note: Do we really need this ::Val{out}? It can be derived from other context btw

# Third argument is a contraint over a single variable, can be a marginalisation contraint for BP or a moment matching contraint for EP
function rule(::Normal, ::Val{:mean}, ::MarginalisationConstraint, ...) end
function rule(::Normal, ::Val{:mean}, ::MomentMatchingContraint, ...) end

# Fourth argument is for messages within the same q_ab expect given edge index
# Fifth argument is for beliefs other q_ab
function rule(::Normal, ::Val{:mean}, ::MarginalisationConstraint,
    messages::NamedTuple{(:precision, :value}, Tuple{Message{Gamma}, Message{Normal}}}, beliefs::Nothing)
end # BP Rule with marginalisation constraint for GaussianMeanPresicion node over the mean

# here in case of q_ab = f(a)
rule(Normal, Val(:mean), MarginalisationConstraint(), (precision = Gamma(blahblah), value = Normal(blahblah)), nothing) # BP rule
rule(Normal, Val(:mean), MomentMatchingConstraint(), (precision = Gamma(blahblah), value = Normal(blahblah)), nothing) # EP rule

function rule(::Normal, ::Val{:mean}, ::MarginalisationConstraint,
    messages::Nothing, beliefs::NamedTuple{(:precision, :value), Tuple{Belief{Gamma}, Belief{Normal}}})
end # VMP rule

rule(Normal, Val(:mean), MarginalisationConstraint(), nothing, (precision = Belief(blahbla), value = Belief(blahblah))) # VMP rule

# And here is the mixed rule

function rule(::Normal, ::Val{:mean}, ::MC,
    messages::Nothing, beliefs::NamedTuple{(:precision_value), Tuple{Belief{Wishart ??}}})
end # Mixed rule

# here in case of q(a) = q_mp(mean, precision)q(value)
rule(Normal, Val(:value), MarginalisationConstraint(), nothing, (mean_precision = Belief(blahblah))) # mean_precision should be a joint belief

# some complex scenario, like q(o) = q(a, b)q(c, d, e)q(f)

rule(MySuperDuperNode, Val(:d), MarginalisationConstraint(), (c = ..., e = ...), (qb = ..., f = ...))

# Draft for m_aj message computation
m_aj = reduce(*, messages \ j) # pseudo-code, but something like this
