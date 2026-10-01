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
    @test contains(vmp, "undefined: the message rule for Gauss towards :μ")

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

@testitem "result display:a joint over a whole group, and another algorithm's mode" tags = [:base] setup = [ResultShowRules] begin
    using MessagePassingRulesBase
    R = ResultShowRules

    struct WholeGroup end
    struct GroupEP <: AbstractAlgorithm end
    @define_factor_node(node = WholeGroup, type = Deterministic, interfaces = [:out, :in...])
    @define_message_update_rule(node = WholeGroup, target = :out, args = (q[(:in,)]::Vector{Float64},), body = (args) -> sum(args.q[(:in,)]))
    @define_message_update_rule(node = WholeGroup, target = :out, algorithm = GroupEP, args = (m[:in...]::Float64,), body = (args) -> sum(args.m[:in]))

    # The joint over the whole group is drawn as one edge that carried it, not as unused.
    joint = call_message_update_rule(WholeGroup, :out; clusters = ((:in,) => [1.0, 2.0],))
    @test contains(R.plain(joint), "in…  ┄┄▶  q") && !contains(R.plain(joint), "unused")
    # Under another algorithm the card names the inputs, not an inference scheme.
    ep = call_message_update_rule(WholeGroup, :out; m = (in = (1.0, 2.0),), algorithm = GroupEP())
    @test contains(R.plain(ep), "· messages") && !contains(R.plain(ep), "belief propagation")

    # A joint given in `q` is pointed to `clusters`.
    err = try
        call_message_update_rule(WholeGroup, :out; q = (in = [1.0, 2.0],))
    catch e
        e
    end
    @test contains(sprint(showerror, err), "note: a rule below takes the joint marginal over :in; a call passes a joint as `clusters = ((:in,) => q,)`")

    # A mean-field call of a shape a rule takes, whose types do not fit, meant its singles: no note.
    struct Paired2 end
    @define_factor_node(node = Paired2, type = Stochastic, interfaces = [:out, :μ, :τ])
    @define_message_update_rule(node = Paired2, target = :τ, args = (q[:out]::Float64, q[:μ]::Float64), body = (args) -> 1.0)
    @define_message_update_rule(node = Paired2, target = :τ, args = (q[:out, :μ]::Tuple,), body = (args) -> 1.0)
    mismatch = try
        call_message_update_rule(Paired2, :τ; q = (out = 1, μ = 2.0))
    catch e
        e
    end
    @test contains(sprint(showerror, mismatch), "type mismatch") && !contains(sprint(showerror, mismatch), "note: a rule below takes the joint")
end

@testitem "result display:a marginal rule reads its members and computes their joint" tags = [:base] setup = [ResultShowRules] begin
    using MessagePassingRulesBase
    R = ResultShowRules
    struct Paired end
    @define_factor_node(node = Paired, type = Stochastic, interfaces = [:out, :μ, :v])
    @define_marginal_update_rule(node = Paired, target = (:out, :μ), args = (m[:out]::Float64, m[:μ]::Float64, q[:v]::Float64), body = (args) -> (args.m[:out], args.m[:μ]))
    result = call_marginal_update_rule(Paired, (:out, :μ); m = (out = 1.0, μ = 2.0), q = (v = 3.0,))
    text = R.plain(result)
    @test contains(text, "out        ──▶  m  1.0") && contains(text, "μ          ──▶  m  2.0")
    @test contains(text, "q(out, μ)  ◀══  target  (1.0, 2.0)") && count("(1.0, 2.0)", text) == 2
    @test !contains(text, "its own edge")
    card = R.html(result)
    @test count("class=\"edge target\"", card) == 1 && count("class=\"edge message\"", card) == 2
end

@testitem "display:generated modules are dropped from printed names" tags = [:base] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: prettify_modules, list_rules

    @test prettify_modules("Main.var\"__atexample__named__first-node\".Gaussian") == "Gaussian"
    @test prettify_modules("Tuple{Main.var\"##TestItem#12\".Gauss, Float64}") == "Tuple{Gauss, Float64}"
    # A package's module, and one a user defines in `Main`, stay.
    @test prettify_modules("q[:v]::BayesBase.PointMass") == "q[:v]::BayesBase.PointMass"
    @test prettify_modules("Main.MyPackage.Node") == "Main.MyPackage.Node"

    # A node and a type declared in a module named as Documenter names its sandboxes.
    sandbox = Core.eval(Main, :(module var"__atexample__named__prettify" end))
    Core.eval(sandbox, :(using MessagePassingRulesBase))
    Core.eval(
        sandbox, quote
            struct Gauss end
            struct Shift end
            @define_factor_node(node = Shift, type = Deterministic, interfaces = [:out, :in])
            @define_message_update_rule(node = Shift, target = :out, args = (m[:in]::Gauss,), body = (args) -> args.m[:in])
        end
    )
    spec = only(list_rules(sandbox.Shift, :out))
    for text in (sprint(show, MIME"text/plain"(), spec), sprint(show, MIME"text/html"(), spec), sprint(show, MIME"text/plain"(), MessagePassingRulesBase.nodespec(sandbox.Shift)))
        @test !contains(text, "__atexample__")
    end
    @test contains(sprint(show, MIME"text/plain"(), spec), "m[:in]::Gauss")
    err = try
        call_message_update_rule(sandbox.Shift, :out; m = (in = 1.0,))
    catch e
        e
    end
    @test !contains(sprint(showerror, err), "__atexample__")
end

@testitem "drawing: a standalone SVG of each drawable" tags = [:base] setup = [ResultShowRules, DependencyNodes] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: drawing, Drawing, nodespec, list_rules, dependencies_spec, SVG_LIGHT
    R, D = ResultShowRules, DependencyNodes
    svg(d) = sprint(show, MIME"image/svg+xml"(), d)
    balanced(text, tag) = count("<$tag", text) == count("</$tag>", text) + count(r"<" * tag * r"\b[^>]*/>", text)

    drawings = (
        node = drawing(nodespec(R.Gauss)),
        rule = drawing(only(list_rules(R.Gauss, :μ))),
        result = drawing(call_message_update_rule(R.Gauss, :out; m = (μ = 1.0, τ = 2.0))),
        dependencies = drawing(dependencies_spec(D.Transition, D.TransitionVMP())),
    )
    for (name, d) in pairs(drawings)
        @testset "$name" begin
            @test d isa Drawing
            @test showable(MIME"image/svg+xml"(), d) && !showable(MIME"text/html"(), d)
            text = svg(d)
            # A document on its own: the SVG namespace, its colours on its elements, no stylesheet.
            @test startswith(text, "<svg xmlns=\"http://www.w3.org/2000/svg\"") && endswith(text, "</svg>")
            @test !contains(text, "<style") && !contains(text, "var(--") && !contains(text, "<script") && !contains(text, "mprb-card")
            @test all(tag -> balanced(text, tag), ("svg", "text", "marker", "defs"))
            @test all(m -> contains(m.match, "stroke=") || contains(m.match, "fill="), eachmatch(r"<(?:line|path|rect)\b[^>]*>", text))
            @test contains(text, "<rect class=\"node\"") && contains(text, "fill=\"$(SVG_LIGHT.bg)\"")
            # Every arrow points at a marker the document defines.
            for m in eachmatch(r"url\(#([^)]+)\)", text)
                @test contains(text, "id=\"$(m.captures[1])\"")
            end
            # Its size and what it shows, in the terminal.
            @test contains(text, "width=\"$(d.width)\"") && contains(text, "height=\"$(d.height)\"")
            @test startswith(sprint(show, MIME"text/plain"(), d), "Drawing of ") && contains(sprint(show, MIME"text/plain"(), d), "$(d.width)×$(d.height) SVG")
        end
    end

    # What each draws, in the light palette.
    @test count("class=\"edge interface\"", svg(drawings.node)) == 4 && contains(svg(drawings.node), ">stochastic</text>")
    @test !contains(svg(drawings.node), "<tspan")
    rule = svg(drawings.rule)
    @test count("class=\"edge marginal\"", rule) == 2 && count("class=\"edge target\"", rule) == 1
    @test contains(rule, "stroke=\"$(SVG_LIGHT.marginal)\"") && contains(rule, "stroke=\"$(SVG_LIGHT.target)\"") && contains(rule, "stroke-dasharray=\"5 3\"")
    result = svg(drawings.result)
    @test count("class=\"edge message\"", result) == 2 && contains(result, "stroke=\"$(SVG_LIGHT.message)\"") && contains(result, "class=\"edge unused\"")
    @test contains(result, "─▶ message m") && contains(result, "━▶ target")
    # The dependencies: one node per target, placed in the document, and the legend under them.
    dependencies = svg(drawings.dependencies)
    targets = length(dependencies_spec(D.Transition, D.TransitionVMP()).targets)
    @test count("<svg class=\"mprb-node\"", dependencies) == targets
    @test count(r"<svg class=\"mprb-node\"[^>]* x=\"\d+\" y=\"\d+\"", dependencies) == targets
    @test count("<defs>", dependencies) == 1 && contains(dependencies, "the default scheme's inputs")

    # `write` saves exactly the markup.
    path = tempname() * ".svg"
    @test write(path, drawings.dependencies) == ncodeunits(dependencies)
    @test read(path, String) == dependencies
    rm(path)

    # Every drawing's markers are its own.
    ids = [match(r"id=\"(mprb-drawing-\d+)-target\"", svg(drawing(only(list_rules(R.Gauss, :μ))))).captures[1] for _ in 1:2]
    @test ids[1] != ids[2]

    # The cards are as before: themed by their stylesheet, the drawing inside without a namespace.
    card = R.html(nodespec(R.Gauss))
    @test contains(card, "<style>") && contains(card, "--mprb-target:$(SVG_LIGHT.target)") && !contains(card, "xmlns")
end
