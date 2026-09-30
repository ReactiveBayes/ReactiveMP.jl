# The pieces the rich displays share: the card that frames them in notebooks and documentation,
# and a node drawn as an SVG, a box with labelled edges. Dependency-free, hand-written markup, so
# a card renders anywhere HTML does, with no script.

# A node by its own name, without the module it is defined in or its parameters:
# `NormalMeanVariance`, `+`. A type is named as it prints, so an alias keeps its name,
# `Categorical` rather than the `DiscreteNonParametric` it stands for.
node_name(node::Function) = string(nameof(node))
node_name(node::Type) = String(last(split(first(split(sprint(print, node), '{')), '.')))
node_name(node) = string(node)

# A group's member as it is written in a drawing, `m[2]`.
member_label(member::Symbol) = string(member)
member_label((group, k)::Tuple) = "$group[$k]"

# A rule's file as its directory and name, which is what tells rules apart.
short_path(file) = (parts = splitpath(String(file)); joinpath(parts[max(end - 1, 1):end]...))

html_escape(x) = replace(string(x), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")
html_code(value) = "<code>" * html_escape(value) * "</code>"

function html_rows(io::IO, rows)
    print(io, "<table>")
    for (key, value) in rows
        print(io, "<tr><th>", html_escape(key), "</th><td>", value, "</td></tr>")
    end
    print(io, "</table>")
    return nothing
end

# Distinct ids for the SVG markers of every card on a page.
const HTML_CARD_COUNTER = Ref(0)

# The palette, light by default. Dark follows the reader's system outside Documenter, and
# Documenter's own theme inside it, whose dark themes mark the root element with a class.
const HTML_CARD_DARK = "--mprb-fg:#e6edf3;--mprb-muted:#8d96a0;--mprb-bg:#0d1117;--mprb-border:#30363d;--mprb-target:#3fb950;--mprb-message:#58a6ff;--mprb-marginal:#bc8cff;--mprb-warn:#d29922"
const HTML_CARD_STYLE = """
.mprb-card{--mprb-fg:#1f2328;--mprb-muted:#6e7781;--mprb-bg:#ffffff;--mprb-border:#d0d7de;--mprb-target:#1a7f37;--mprb-message:#0969da;--mprb-marginal:#8250df;--mprb-warn:#9a6700;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:13px;color:var(--mprb-fg);background:var(--mprb-bg);border:1px solid var(--mprb-border);border-radius:8px;padding:12px 14px;margin:6px 0;max-width:960px}
@media (prefers-color-scheme: dark){:root:not(:has(#documenter)) .mprb-card{$(HTML_CARD_DARK)}}
html.theme--documenter-dark .mprb-card,html.theme--catppuccin-frappe .mprb-card,html.theme--catppuccin-macchiato .mprb-card,html.theme--catppuccin-mocha .mprb-card{$(HTML_CARD_DARK)}
.mprb-card .mprb-head{font-weight:600;margin-bottom:8px}.mprb-card .mprb-mode{color:var(--mprb-muted);font-weight:400;margin-left:6px}
.mprb-card .mprb-body{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}.mprb-card .mprb-sections{flex:1;min-width:280px}
.mprb-card .mprb-figure{margin:0}.mprb-card .mprb-figure figcaption{color:var(--mprb-muted);font-size:12px}
.mprb-card details{margin:4px 0}.mprb-card summary{cursor:pointer;font-weight:600}
.mprb-card table{border-collapse:collapse;margin:4px 0 6px 0;color:inherit;font-size:inherit;font-family:inherit;background:transparent}.mprb-card td,.mprb-card th{padding:2px 8px 2px 0;text-align:left;vertical-align:top;border:none;background:transparent}
.mprb-card th{color:var(--mprb-muted);font-weight:500}.mprb-card code,.mprb-card pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;background:transparent;color:inherit;padding:0}
.mprb-card pre{white-space:pre-wrap;margin:2px 0;border:none}.mprb-card .mprb-ok{color:var(--mprb-target)}.mprb-card .mprb-no{color:var(--mprb-warn)}.mprb-card .mprb-undefined{color:var(--mprb-warn)}
.mprb-card svg{max-width:100%;height:auto}
.mprb-card svg text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;fill:var(--mprb-fg);stroke:none}
.mprb-card svg text.note,.mprb-card svg tspan.note,.mprb-card svg text.kind{fill:var(--mprb-muted);font-size:11px}
.mprb-card svg text.target{fill:var(--mprb-target)}.mprb-card svg marker path{stroke:none}
.mprb-card svg marker path.target{fill:var(--mprb-target)}.mprb-card svg marker path.message{fill:var(--mprb-message)}
.mprb-card svg marker path.marginal,.mprb-card svg marker path.joint{fill:var(--mprb-marginal)}
.mprb-card svg line.edge{stroke-width:1.6;fill:none}.mprb-card svg line.target{stroke:var(--mprb-target);stroke-width:2.4}
.mprb-card svg line.message{stroke:var(--mprb-message)}.mprb-card svg line.marginal,.mprb-card svg line.joint{stroke:var(--mprb-marginal);stroke-dasharray:5 3}
.mprb-card svg line.interface{stroke:var(--mprb-fg)}.mprb-card svg line.default{stroke:var(--mprb-muted);stroke-dasharray:1 3}
.mprb-card svg line.unused{stroke:var(--mprb-border);stroke-dasharray:2 3}.mprb-card svg text.unused,.mprb-card svg text.default{fill:var(--mprb-muted)}
.mprb-card svg text.message{fill:var(--mprb-message)}.mprb-card svg text.marginal,.mprb-card svg text.joint{fill:var(--mprb-marginal)}
.mprb-card svg .node{fill:var(--mprb-bg);stroke:var(--mprb-fg);stroke-width:1.6}
.mprb-card .mprb-legend span{margin-right:12px;font-size:12px}.mprb-card .mprb-legend .message{color:var(--mprb-message)}.mprb-card .mprb-legend .marginal{color:var(--mprb-marginal)}.mprb-card .mprb-legend .target{color:var(--mprb-target)}
.mprb-card .mprb-grid{display:flex;flex-wrap:wrap;gap:10px 24px;align-items:flex-start}
"""

# Open a card with its heading and an optional muted note beside it; returns the card's id, which
# prefixes the ids of the markers its drawings define. Close it with `close_card`.
function open_card(io::IO, heading, note = "")
    id = "mprb-" * string(HTML_CARD_COUNTER[] += 1)
    print(io, "<div class=\"mprb-card\" id=\"", id, "\"><style>", HTML_CARD_STYLE, "</style>")
    print(io, "<div class=\"mprb-head\">", html_escape(heading))
    isempty(note) || print(io, "<span class=\"mprb-mode\">", html_escape(note), "</span>")
    print(io, "</div>")
    return id
end
close_card(io::IO) = print(io, "</div>")

# What an edge of a drawn node is, which sets its colour, its dash and its arrow:
#   `:target`    the edge a rule computes for, an arrow out of the node;
#   `:message`   an input taken as a message, `m`, an arrow in;
#   `:marginal`  an input taken as a marginal, `q`, a dashed arrow in;
#   `:joint`     a member of a joint marginal, as `:marginal`;
#   `:default`   the inputs the default scheme delivers, dotted;
#   `:interface` an edge of a declaration, a plain line;
#   `:unused`    an edge a call leaves out, greyed.
struct SvgEdge
    label::String
    role::Symbol
    note::String
end
SvgEdge(label, role) = SvgEdge(string(label), role, "")

role_class(role) = string(role)
has_arrow(role) = role in (:target, :message, :marginal, :joint)

# The width of a label in the SVG's 12px monospace font, a note in 11px after it.
label_width(edge::SvgEdge) = 7.3 * length(edge.label) + (isempty(edge.note) ? 0 : 6.7 * (length(edge.note) + 3))

# A node drawn as a box named `name`, with `left` edges coming in and `right` edges going out, each
# edge a line with its label at the far end. `kind`, `"stochastic"` or `"deterministic"`, is
# written under the name when given. `id` is the card's, for the arrow markers.
function svg_node(io::IO, id, name; left::Vector{SvgEdge} = SvgEdge[], right::Vector{SvgEdge} = SvgEdge[], kind = "", aria = name, stub = 60)
    rows = max(length(left), length(right), 1)
    height = max(30 * rows + 30, isempty(kind) ? 60 : 76)
    leftw = round(Int, maximum(label_width, left; init = 0.0)) + 10
    rightw = isempty(right) ? 0 : round(Int, maximum(label_width, right; init = 0.0)) + 14
    box_width = max(60, round(Int, 8 * max(length(name), length(kind) * 0.9)) + 20)
    box_x = leftw + (isempty(left) ? 0 : stub)
    width = box_x + box_width + (isempty(right) ? 4 : stub + rightw)
    box_y, box_h = 15, height - 30
    print(io, "<svg class=\"mprb-node\" role=\"img\" aria-label=\"", html_escape(aria), "\" width=\"", width, "\" height=\"", height, "\" viewBox=\"0 0 ", width, " ", height, "\">")
    print(io, "<defs>")
    for role in (:target, :message, :marginal, :joint)
        print(io, "<marker id=\"", id, "-", role, "\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerUnits=\"userSpaceOnUse\" markerWidth=\"9\" markerHeight=\"9\" orient=\"auto-start-reverse\">")
        print(io, "<path d=\"M0,0 L10,5 L0,10 z\" class=\"", role, "\"/></marker>")
    end
    print(io, "</defs>")
    y_of(i, n) = box_y + box_h * i / (n + 1)
    for (i, edge) in enumerate(left)
        y = y_of(i, length(left))
        marker = has_arrow(edge.role) ? " marker-end=\"url(#$id-$(edge.role))\"" : ""
        print(io, "<line class=\"edge ", role_class(edge.role), "\" x1=\"", leftw, "\" y1=\"", y, "\" x2=\"", box_x, "\" y2=\"", y, "\"", marker, "/>")
        print(io, "<text class=\"", role_class(edge.role), "\" x=\"", leftw - 6, "\" y=\"", y + 4, "\" text-anchor=\"end\">", html_escape(edge.label))
        isempty(edge.note) || print(io, "<tspan class=\"note\"> (", html_escape(edge.note), ")</tspan>")
        print(io, "</text>")
    end
    for (i, edge) in enumerate(right)
        y = y_of(i, length(right))
        x1, x2 = box_x + box_width, box_x + box_width + stub
        marker = has_arrow(edge.role) ? " marker-end=\"url(#$id-$(edge.role))\"" : ""
        weight = edge.role === :target ? " font-weight=\"bold\"" : ""
        print(io, "<line class=\"edge ", role_class(edge.role), "\" x1=\"", x1, "\" y1=\"", y, "\" x2=\"", x2, "\" y2=\"", y, "\"", marker, "/>")
        print(io, "<text class=\"", role_class(edge.role), "\" x=\"", x2 + 8, "\" y=\"", y + 4, "\"", weight, ">", html_escape(edge.label))
        isempty(edge.note) || print(io, "<tspan class=\"note\"> (", html_escape(edge.note), ")</tspan>")
        print(io, "</text>")
    end
    print(io, "<rect class=\"node\" x=\"", box_x, "\" y=\"", box_y, "\" width=\"", box_width, "\" height=\"", box_h, "\" rx=\"4\"/>")
    centre, middle = box_x + box_width / 2, box_y + box_h / 2
    if isempty(kind)
        print(io, "<text x=\"", centre, "\" y=\"", middle + 4, "\" text-anchor=\"middle\">", html_escape(name), "</text>")
    else
        print(io, "<text x=\"", centre, "\" y=\"", middle - 2, "\" text-anchor=\"middle\">", html_escape(name), "</text>")
        print(io, "<text class=\"kind\" x=\"", centre, "\" y=\"", middle + 13, "\" text-anchor=\"middle\">", html_escape(kind), "</text>")
    end
    print(io, "</svg>")
    return nothing
end

# The key to the edge colours, under a drawing.
function html_legend(io::IO, roles)
    print(io, "<div class=\"mprb-legend\">")
    :target in roles && print(io, "<span class=\"target\">━▶ target</span>")
    :message in roles && print(io, "<span class=\"message\">─▶ message m</span>")
    (:marginal in roles || :joint in roles) && print(io, "<span class=\"marginal\">┄▶ marginal q</span>")
    :default in roles && print(io, "<span class=\"mprb-mode\">··· the default scheme's inputs</span>")
    print(io, "</div>")
    return nothing
end
