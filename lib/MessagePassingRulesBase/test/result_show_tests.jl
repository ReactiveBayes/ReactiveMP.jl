@testmodule ResultShowRules begin
    using MessagePassingRulesBase

    struct Gauss end
    @define_factor_node(node = Gauss, type = Stochastic, interfaces = [:out, :μ, :τ, :w...])
    @define_message_update_rule(node = Gauss, target = :out, args = (m[:μ]::Float64, m[:τ]::Float64), logscale = 0, body = (args) -> args.m[:μ] + args.m[:τ])
    @define_message_update_rule(node = Gauss, target = :out, args = (m[:μ]::Int, m[:τ]::Int), body = (args) -> args.m[:μ] + args.m[:τ])
    @define_message_update_rule(node = Gauss, target = :μ, args = (q[:out]::Float64, q[:τ]::Float64), body = (args) -> args.q[:out] * args.q[:τ])
    @define_message_update_rule(node = Gauss, target = :τ, args = (q[:out, :μ]::Tuple,), body = (args) -> sum(args.q[:out, :μ]))
    @define_average_energy(node = Gauss, args = (q[:out]::Float64,), body = (args) -> args.q[:out])

    plain(x; kwargs...) = sprint(show, MIME"text/plain"(), x; context = (kwargs...,))
    html(x) = sprint(show, MIME"text/html"(), x)
end

@testitem "result display:text/plain" tags = [:base] setup = [ResultShowRules] begin
    using MessagePassingRulesBase
    R = ResultShowRules

    bp = call_message_update_rule(R.Gauss, :out; m = (μ = 1.0, τ = 2.0))
    text = R.plain(bp)
    @test startswith(text, "RuleResult  message of Gauss towards :out")
    @test contains(text, "belief propagation")
    # One line per edge: the target, the messages with their values, the unused group.
    @test contains(text, "out  ◀══  target  3.0")
    @test contains(text, "μ    ──▶  m  1.0")
    @test contains(text, "τ    ──▶  m  2.0")
    @test contains(text, "w…")
    @test contains(text, "logscale   0  (declared)")
    @test contains(text, "@ test/result_show_tests.jl:")
    @test contains(text, "and 1 other rule for this target")
    # Colour only where the stream asks for it; the compact form is the one-line one.
    @test !contains(text, "\e[")
    @test contains(R.plain(bp; color = true), "\e[")
    @test R.plain(bp; compact = true) == repr(bp)

    vmp = R.plain(call_message_update_rule(R.Gauss, :μ; q = (out = 1.0, τ = 2.0)))
    @test contains(vmp, "variational") && contains(vmp, "out  ┄┄▶  q  1.0")
    @test contains(vmp, "undefined: the message rule for $(R.Gauss) towards :μ")

    # A joint arrives once; its other members name it.
    joint = R.plain(call_message_update_rule(R.Gauss, :τ; clusters = ((:out, :μ) => (1.0, 2.0),)))
    @test contains(joint, "out  ┄┄▶  q  (1.0, 2.0)  (q[:out, :μ])")
    @test contains(joint, "μ    ┄┄▶  q  (in q[:out, :μ])")

    energy = R.plain(call_average_energy(R.Gauss; q = (out = 1.0,)))
    @test contains(energy, "average energy of Gauss") && !contains(energy, "◀══")
end

@testitem "result display:text/html" tags = [:base] setup = [ResultShowRules] begin
    using MessagePassingRulesBase
    R = ResultShowRules

    bp = call_message_update_rule(R.Gauss, :out; m = (μ = 1.0, τ = 2.0))
    @test showable(MIME"text/html"(), bp)
    card = R.html(bp)
    # Self-contained: one card with its own style, themed, no script.
    @test startswith(card, "<div class=\"mprb-card\"") && endswith(card, "</div>")
    @test contains(card, "<style>") && contains(card, "prefers-color-scheme: dark") && !contains(card, "<script")
    # The node drawn: its box, the target's arrow out, the messages' arrows in, the unused group.
    @test count("<svg", card) == 1 && contains(card, "<rect class=\"node\"")
    @test contains(card, "class=\"edge target\"") && count("class=\"edge message\"", card) == 2
    @test contains(card, "class=\"edge unused\"") && contains(card, ">w…</text>")
    # The sections, each balanced.
    @test count("<details", card) == count("</details>", card) == 4
    for section in ("Result", "Inputs", "Rule", "Other rules for this target (1)")
        @test contains(card, "<summary>$section</summary>")
    end
    @test count("<table>", card) == count("</table>", card)
    # Every card's markers are its own.
    ids = [match(r"id=\"(mprb-\d+)\"", R.html(bp)).captures[1] for _ in 1:2]
    @test ids[1] != ids[2]

    vmp = R.html(call_message_update_rule(R.Gauss, :μ; q = (out = 1.0, τ = 2.0)))
    @test count("class=\"edge marginal\"", vmp) == 2 && contains(vmp, "mprb-undefined")
    @test contains(R.html(call_message_update_rule(R.Gauss, :τ; clusters = ((:out, :μ) => (1.0, 2.0),))), "class=\"edge joint\"")
    # Values are escaped.
    @test !contains(R.html(call_message_update_rule(R.Gauss, :out; m = (μ = 1, τ = 2))), "<Int")
end

@testitem "spec display:text/html" tags = [:base] setup = [ResultShowRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: nodespec, list_rules, RuleNotFound
    R = ResultShowRules

    # A declaration draws its node: the output on the right, the other interfaces on the left,
    # a group by its name, then the interfaces and the declaration's fields.
    node = R.html(nodespec(R.Gauss))
    @test startswith(node, "<div class=\"mprb-card\"") && endswith(node, "</div>") && !contains(node, "<script")
    @test count("<svg", node) == 1 && contains(node, "<rect class=\"node\"") && contains(node, ">stochastic</text>")
    @test count("class=\"edge interface\"", node) == 4 && contains(node, ">w…</text>")
    @test contains(node, "a group of any number of members") && contains(node, "default algorithm")
    @test count("<table>", node) == count("</table>", node)

    # A rule draws what it consumes, coloured by how, and what it computes.
    bp = R.html(only(filter(s -> s.inputs[1].type === Float64, list_rules(R.Gauss, :out))))
    @test count("class=\"edge message\"", bp) == 2 && count("class=\"edge target\"", bp) == 1
    @test contains(bp, "m[:μ]") && contains(bp, "<pre>") && contains(bp, "class=\"mprb-legend\"")
    vmp = R.html(only(list_rules(R.Gauss, :μ)))
    @test count("class=\"edge marginal\"", vmp) == 2
    joint = R.html(only(list_rules(R.Gauss, :τ)))
    @test contains(joint, "class=\"edge joint\"") && contains(joint, "q[:out, :μ]")
    energy = R.html(only(filter(s -> s.kind === :average_energy, list_rules(R.Gauss))))
    @test !contains(energy, "class=\"edge target\"")

    # The pieces of a declaration show themselves as they are written.
    @test sprint(show, nodespec(R.Gauss).interfaces[4]) == "w..."
    @test sprint(show, only(list_rules(R.Gauss, :μ)).inputs[1]) == "q[:out]::Float64"
    missing_rule = RuleNotFound(:message, R.Gauss, MessagePassingRulesBase.Target(:out), DefaultAlgorithm(), nothing)
    @test startswith(sprint(show, missing_rule), "RuleNotFound(message rule for ") && contains(sprint(show, missing_rule), "towards :out under ")
end

@testitem "result display:an input on the target's own edge" tags = [:base] setup = [ResultShowRules] begin
    using MessagePassingRulesBase
    R = ResultShowRules

    # A rule towards `in` that reads the message on `in` itself shows it beside the target.
    struct OwnEdge end
    @define_factor_node(node = OwnEdge, type = Stochastic, interfaces = [:out, :in])
    @define_message_update_rule(node = OwnEdge, target = :in, args = (m[:out]::Float64, m[:in]::Float64), body = (args) -> args.m[:out] - args.m[:in])
    result = call_message_update_rule(OwnEdge, :in; m = (out = 3.0, in = 1.0))
    text = R.plain(result)
    @test contains(text, "in   ◀══  target  2.0") && contains(text, "in   ──▶  m  1.0  (its own edge)")
    card = R.html(result)
    @test count("class=\"edge message\"", card) == 2 && contains(card, "its own edge")
end
