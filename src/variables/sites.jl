## The sites of a random variable: cavity propagation

"""
    ReactiveMP.compute_site(policy, variable, index, message, cavity, previous) -> (site, outcome)

The site of edge `index` of a [`RandomVariable`](@ref) activated with a site policy (see
[`RandomVariableActivationOptions`](@ref)): what replaces the edge's inbound message in the
variable's products, its outbound messages and its marginal. A site is the inbound message
projected against its cavity, `proj(message · cavity) / cavity`, the cavity being the product of
the other edges' sites, the variable's outbound message on the same edge: expectation
propagation, or Laplace propagation, where every belief of the variable is a product of sites.

- `message`: the inbound message's data;
- `cavity`: the product of the other edges' latest sites, `missing` while none has one;
- `previous`: the `outcome` of the edge's last site, `nothing` at first, where a projection can
  start from.

It returns the site's data and an `outcome`, any value, which the
[`ReactiveMP.AfterSiteComputationEvent`](@ref) carries. A site `missing` means none yet, as while
the cavity is `missing`: the edge then contributes nothing to the products until its next inbound
message. A site `nothing` means that the edge's site has not changed: the last one is sent again.

The engine knows no projection: an inference package defines a method for its policy. RxInfer's
cavity schemes, `LaplacePropagation()` and `ExpectationPropagation()`, give one.
"""
function compute_site end

"""
    ReactiveMP.seeded_cavity(policy, cavity, initial, waiting, degree) -> cavity

The cavity of a site while `waiting` of the other edges of a variable with `degree` connections
have no site yet, and the variable has an initial message `initial` (`@initialization`'s `μ(x)`):
`cavity` (`missing` while no other edge has a site) times `initial^(waiting / (degree - 1))`.
Every edge's first cavity is then the initial message itself, whatever the degree, and the seed
fades as the edges' own sites arrive. It enters the cavities only: the outbound messages and the
marginal are products of the sites alone.

The default returns `cavity` as it is; an inference package defines a method for its policy, as
for [`ReactiveMP.compute_site`](@ref).
"""
seeded_cavity(policy, cavity, initial, waiting, degree) = cavity

# What the sites of a variable share: the initial message, if any, which edges have a site, and
# the edges themselves, so that a static one can be projected again (`refresh_static!`)
mutable struct SiteState
    initial::Any
    present::Vector{Bool}
    count::Int
    mappings::Vector{Any}
    sites::Vector{MessageObservable{AbstractMessage}}
    # the last edge, by index, whose inbound message is not clamped: its site starts a round
    last_dynamic::Int
end

SiteState(sites::Vector{MessageObservable{AbstractMessage}}) =
    SiteState(nothing, fill(false, length(sites)), 0, Any[], sites, 0)

function mark!(state::SiteState, index, present::Bool)
    state.present[index] == present && return nothing
    state.present[index] = present
    state.count += present ? 1 : -1
    return nothing
end

# The cavity from the chain, and, while other edges have no site, the initial message for each
function seeded(mapping, cavity)
    state = mapping.state
    isnothing(state.initial) && return cavity
    d = length(state.present)
    waiting = d - state.count - (state.present[mapping.index] ? 0 : 1)
    waiting == 0 && return cavity
    return Message(seeded_cavity(mapping.policy, getdata(cavity), state.initial, waiting, d), false, true)
end

# The site of one edge, computed as the edge's inbound message arrives, against the cavity the
# variable's equality chain holds then: the product of the other edges' latest sites, from the
# chain's cached partial products. The sites are the chain's inputs and only an inbound message
# triggers a site, so the streams have no cycle, and the chain consumes the sites as it would the
# inbound messages: the outbound messages and the marginal are sent once each has new inputs. With
# the messages of an iteration arriving one by one, each site against the latest others, an
# iteration is a sweep of expectation propagation over the sites.
#
# Other wirings were tried and rejected. A site computed from the latest inbound and outbound
# messages of its edge (`combineLatestUpdates`) deadlocks: site k waits for a new outbound message
# k, which waits for new sites on the other edges, which wait for theirs. Firing on the inbound
# message and sampling the outbound one (`PushEachBut`) reads a cavity about an iteration old, a
# parallel schedule: 264 to 524 iterations where the sweep takes 4 to 12. And a separate stage
# that kept the sites and sent each product as a site changed grew as d³ per iteration in the
# number of connections d: one iteration of a logistic regression with 1000 observations took
# 117 s, against 0.34 s through the chain. A site is sent even when it has not changed: holding it
# back froze the consumers that fire once each of their inputs is new, such as a rule with two
# inputs.
#
# A raw message that is clamped, from a node whose inputs are all constants, arrives once, and its
# stream completes: its site, static, would keep the cavity of that moment, or wait for ever where
# it had none. So each round, as the last dynamic edge's message arrives, every static site is
# projected again against its cavity then (`refresh_static!`), and the chain and the marginal read
# each site's latest.
mutable struct SiteMapping{P, V, C, H}
    policy::P
    variable::V
    index::Int
    callbacks::C
    chain::H
    state::SiteState
    outcome::Any
    site::Message
    inbound::Any   # the last inbound message, `nothing` before the first
    projected::Bool # whether the site is a projection, or waits for one, rather than the message
end

SiteMapping(policy, variable, index, callbacks, chain, state = SiteState(chain.inputmsgs)) =
    SiteMapping(policy, variable, index, callbacks, chain, state, nothing, Message(missing, false, true), nothing, false)

function (mapping::SiteMapping)(inbound)
    message = as_message(inbound)
    state, index = mapping.state, mapping.index
    if !is_clamped(message)
        state.last_dynamic = max(state.last_dynamic, index)
        index == state.last_dynamic && refresh_static!(state, index)
    end
    return compute_site!(mapping, message)
end

function compute_site!(mapping::SiteMapping, message::Message)
    chain, index, callbacks = mapping.chain, mapping.index, mapping.callbacks
    cavity = seeded(mapping, ChainOutboundMapping(index, chain)(nothing))
    span_id = generate_span_id(callbacks)
    @invoke_callback(callbacks, BeforeSiteComputationEvent(mapping.variable, index, message, cavity, span_id))
    site, outcome = compute_site(mapping.policy, mapping.variable, index, getdata(message), getdata(cavity), mapping.outcome)
    @invoke_callback(callbacks, AfterSiteComputationEvent(mapping.variable, index, message, cavity, site, outcome, span_id))
    isnothing(outcome) || (mapping.outcome = outcome)
    mapping.inbound = message
    # the inbound message's own flags where it is its own site; a projection is clamped only
    # where both are, and initial where both are clamped or initial, as a rule's output is
    mapping.site = if isnothing(site)
        mapping.site
    elseif ismissing(site)
        mapping.projected = true
        Message(missing, false, true)
    elseif site === getdata(message)
        mapping.projected = false
        Message(site, is_clamped(message), is_initial(message), getannotations(message), nothing)
    else
        mapping.projected = true
        clamped = is_clamped(message) && is_clamped(cavity)
        Message(site, clamped, !clamped && is_clamped_or_initial(message) && is_clamped_or_initial(cavity))
    end
    mark!(mapping.state, index, has_data(mapping.site))
    # the chain's products that hold this edge's site are stale, whichever of its streams have
    # subscribers
    ChainInvalidationCallback(index, chain)(nothing)
    return mapping.site
end

latest_sites(sites, messages) = map((site, message) -> something(getrecent(site), message), sites, messages)

# Every static site, a projection of a clamped message, projected again against its cavity now,
# and sent as an initial message
function refresh_static!(state::SiteState, trigger)
    for (k, other) in enumerate(state.mappings)
        (k == trigger || isnothing(other.inbound) || !other.projected || !is_clamped(other.inbound)) && continue
        site = compute_site!(other, other.inbound)
        other.site = Message(getdata(site), false, true)
        next!(state.sites[k].subject, other.site)
    end
    return nothing
end

Base.map(::Type{Message}, mapping::M) where {M <: SiteMapping} =
    Rocket.MapOperator{Message, M}(mapping)

has_data(value) = !ismissing(getdata(value))

# The sites of a variable with `d > 1` connections: the equality chain over them, with the
# variable's outbound messages and marginal as its products. A product still `missing`, every
# site it holds waiting, is not sent.
function activate_sites!(randomvar, options::RandomVariableActivationOptions, d)
    callbacks = options.prod_context_for_marginal_computation.callbacks
    sites = [MessageObservable(AbstractMessage) for _ in 1:d]
    chain = EqualityChain(sites, options.stream_postprocessor, randomvar, options.prod_context_for_message_computation)
    state = SiteState(sites)
    randomvar.sites = state
    for index in 1:d
        mapping = SiteMapping(options.site_policy, randomvar, index, callbacks, chain, state)
        push!(state.mappings, mapping)
        connect!(sites[index], randomvar.input_messages[index] |> map(Message, mapping))
    end
    products = [MessageObservable(Message) for _ in 1:d]
    initialize!(chain, products)
    foreach(index -> connect!(randomvar.output_messages[index], products[index] |> filter(has_data)), 1:d)
    # each site as the chain reads it, its latest: a static site's stream completes with its
    # message's, and is projected again after that (`refresh_static!`)
    marginals = collectLatest(AbstractMessage, Marginal, sites, (messages) -> _compute_marginal_from_messages(randomvar, options, latest_sites(sites, messages)), reset_vstatus)
    set_stream_of_marginals!(randomvar, postprocess_stream_of_marginals(options.stream_postprocessor, marginals |> filter(has_data)))
    return nothing
end
