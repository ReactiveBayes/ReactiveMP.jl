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

# The site of one edge, computed as the edge's inbound message arrives, against the cavity the
# variable's equality chain holds then: the product of the other edges' latest sites, from the
# chain's cached partial products. The sites are the chain's inputs and only an inbound message
# triggers a site, so the streams have no cycle, and the chain consumes the sites as it would the
# inbound messages: the outbound messages and the marginal are sent once each has new inputs. With
# the messages of an iteration arriving one by one, each site against the latest others, an
# iteration is a sweep of expectation propagation over the sites.
mutable struct SiteMapping{P, V, C, H}
    policy::P
    variable::V
    index::Int
    callbacks::C
    chain::H
    outcome::Any
    site::Message
end

SiteMapping(policy, variable, index, callbacks, chain) =
    SiteMapping(policy, variable, index, callbacks, chain, nothing, Message(missing, false, true))

function (mapping::SiteMapping)(inbound)
    message = as_message(inbound)
    chain, index, callbacks = mapping.chain, mapping.index, mapping.callbacks
    cavity = ChainOutboundMapping(index, chain)(nothing)
    span_id = generate_span_id(callbacks)
    @invoke_callback(callbacks, BeforeSiteComputationEvent(mapping.variable, index, message, cavity, span_id))
    site, outcome = compute_site(mapping.policy, mapping.variable, index, getdata(message), getdata(cavity), mapping.outcome)
    @invoke_callback(callbacks, AfterSiteComputationEvent(mapping.variable, index, message, cavity, site, outcome, span_id))
    isnothing(outcome) || (mapping.outcome = outcome)
    # the inbound message's own flags where it is its own site; a projection is clamped only
    # where both are, and initial where both are clamped or initial, as a rule's output is
    mapping.site = if isnothing(site)
        mapping.site
    elseif ismissing(site)
        Message(missing, false, true)
    elseif site === getdata(message)
        Message(site, is_clamped(message), is_initial(message), getannotations(message), nothing)
    else
        clamped = is_clamped(message) && is_clamped(cavity)
        Message(site, clamped, !clamped && is_clamped_or_initial(message) && is_clamped_or_initial(cavity))
    end
    # the chain's products that hold this edge's site are stale, whichever of its streams have
    # subscribers
    ChainInvalidationCallback(index, chain)(nothing)
    return mapping.site
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
    for index in 1:d
        mapping = SiteMapping(options.site_policy, randomvar, index, callbacks, chain)
        connect!(sites[index], randomvar.input_messages[index] |> map(Message, mapping))
    end
    products = [MessageObservable(Message) for _ in 1:d]
    initialize!(chain, products)
    foreach(index -> connect!(randomvar.output_messages[index], products[index] |> filter(has_data)), 1:d)
    marginals = collectLatest(AbstractMessage, Marginal, sites, (messages) -> _compute_marginal_from_messages(randomvar, options, messages), reset_vstatus)
    set_stream_of_marginals!(randomvar, postprocess_stream_of_marginals(options.stream_postprocessor, marginals |> filter(has_data)))
    return nothing
end
