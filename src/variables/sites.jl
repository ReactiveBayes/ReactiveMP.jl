## The site stage of a random variable: cavity propagation

"""
    ReactiveMP.compute_site(policy, variable, index, message, cavity, previous) -> (site, outcome)

The site of edge `index` of a [`RandomVariable`](@ref) activated with a site policy (see
[`RandomVariableActivationOptions`](@ref)): what replaces the edge's inbound message in the
variable's products, its outbound messages and its marginal. A site is the inbound message
projected against its cavity, `proj(message · cavity) / cavity`, the cavity being the variable's
outbound message on the same edge, the product of the other edges' sites: expectation
propagation, or Laplace propagation, where every belief of the variable is a product of sites.

- `message`: the inbound message's data;
- `cavity`: the outbound message's data, `missing` while no other edge has a site;
- `previous`: the `outcome` of the edge's last site, `nothing` at first, where a projection can
  start from.

It returns the site's data and an `outcome`, any value, which the
[`ReactiveMP.AfterSiteComputationEvent`](@ref) carries. A site `missing` means none yet, as while
the cavity is `missing`: the edge then contributes nothing to the products, and its site is
computed again once the cavity has a value. A site `nothing` means that the edge's site has not
changed, and nothing is sent.

The engine knows no projection: an inference package defines a method for its policy. RxInfer's
cavity schemes, `LaplacePropagation()` and `ExpectationPropagation()`, give one.
"""
function compute_site end

# The state of a variable's sites. A site is computed when its edge's inbound message arrives,
# against the product of the other sites as they are then; a site still waiting for a cavity,
# when one exists. The variable's outbound messages and its marginal are the products of its
# sites, sent each time a site changes, the outbound message on an edge only when a site of
# another edge changed, and only once it is not `missing`; each keeps its latest value for a
# subscriber that comes later, since the stage computes as messages arrive, not as it is
# subscribed to, as the equality chain does. With the data of an iteration arriving
# one by one, and the products updated at once, each iteration is a sweep of expectation
# propagation over the sites, each against the cavity of the latest sites.
mutable struct SiteStage{P, V, C, M}
    policy::P
    variable::V
    message_context::C
    marginal_context::M
    inbound::Vector{Any}
    sites::Vector{Message}
    outcomes::Vector{Any}
    waiting::Vector{Bool}
    outputs::Vector{Rocket.RecentSubjectInstance{Message, Subject{Message, AsapScheduler, AsapScheduler}}}
    marginal::Rocket.RecentSubjectInstance{Marginal, Subject{Marginal, AsapScheduler, AsapScheduler}}
end

function SiteStage(policy, variable, message_context, marginal_context, d)
    return SiteStage(
        policy, variable, message_context, marginal_context,
        Any[nothing for _ in 1:d],
        Message[Message(missing, false, true) for _ in 1:d],
        Any[nothing for _ in 1:d],
        fill(false, d),
        [RecentSubject(Message) for _ in 1:d],
        RecentSubject(Marginal),
    )
end

Base.length(stage::SiteStage) = length(stage.sites)

# The product of the sites but `skip`'s, as the outbound message on that edge, between the
# product events and with the messages' form constraint; `missing` while every other site is
function sites_product(stage::SiteStage, context, skip)
    others = [stage.sites[i] for i in 1:length(stage) if i != skip]
    isempty(others) && return Message(missing, true, true)
    return compute_product_of_messages(stage.variable, context, others)
end

function on_inbound!(stage::SiteStage, index, message)
    stage.inbound[index] = message
    changed = fill(false, length(stage))
    update_site!(stage, index, changed)
    # sites that waited for a cavity, now that another site may have given one
    progress = true
    while progress
        progress = false
        for i in 1:length(stage)
            stage.waiting[i] || continue
            update_site!(stage, i, changed) && (progress = true)
        end
    end
    emit!(stage, changed)
    return nothing
end

# The site of edge `index`, between the site events; whether it changed
function update_site!(stage::SiteStage, index, changed)
    message = as_message(stage.inbound[index])
    cavity = sites_product(stage, stage.message_context, index)
    callbacks = stage.marginal_context.callbacks
    span_id = generate_span_id(callbacks)
    @invoke_callback(callbacks, BeforeSiteComputationEvent(stage.variable, index, message, cavity, span_id))
    site, outcome = compute_site(stage.policy, stage.variable, index, getdata(message), getdata(cavity), stage.outcomes[index])
    @invoke_callback(callbacks, AfterSiteComputationEvent(stage.variable, index, message, cavity, site, outcome, span_id))
    isnothing(outcome) || (stage.outcomes[index] = outcome)
    stage.waiting[index] = ismissing(site)
    (ismissing(site) || isnothing(site)) && return false
    # the inbound message's own flags where it is its own site; a projection is clamped only
    # where both are, and initial where both are clamped or initial, as a rule's output is
    stage.sites[index] = if site === getdata(message)
        Message(site, is_clamped(message), is_initial(message), getannotations(message), nothing)
    else
        clamped = is_clamped(message) && is_clamped(cavity)
        Message(site, clamped, !clamped && is_clamped_or_initial(message) && is_clamped_or_initial(cavity))
    end
    changed[index] = true
    return true
end

function emit!(stage::SiteStage, changed)
    any(changed) || return nothing
    for j in 1:length(stage)
        any(i -> changed[i] && i != j, 1:length(stage)) || continue
        outbound = sites_product(stage, stage.message_context, j)
        ismissing(getdata(outbound)) || next!(stage.outputs[j], outbound)
    end
    marginal = compute_marginal_from_sites(stage)
    ismissing(getdata(marginal)) || next!(stage.marginal, marginal)
    return nothing
end

function compute_marginal_from_sites(stage::SiteStage)
    context = stage.marginal_context
    span_id = generate_span_id(context.callbacks)
    @invoke_callback(context.callbacks, BeforeMarginalComputationEvent(stage.variable, context, Tuple(stage.sites), span_id))
    result = as_marginal(compute_product_of_messages(stage.variable, context, stage.sites))
    @invoke_callback(context.callbacks, AfterMarginalComputationEvent(stage.variable, context, Tuple(stage.sites), result, span_id))
    return result
end
