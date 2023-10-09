ReactiveMP.jl
=============

*Julia package for reactive message passing Bayesian inference engine on a factor graph.*

!!! note
    This package exports only an inference engine, for the full ecosystem with convenient model and constraints specification we refer user to the [`RxInfer.jl`](https://github.com/biaslab/RxInfer.jl) package and its [documentation](https://biaslab.github.io/RxInfer.jl/stable/).

## Rules


```@example tables
using ReactiveMP, Markdown

function print_rules_table()
    mtds = methods(ReactiveMP.rule)
    Markdown.parse(
        """
        RxInfer supports $(length(mtds)) analytical message passing update rules! Below you can find an overview of these rules:
        
        | Node | Output | Inputs | Meta |
        |:-----|:-------|:-------|:-----|
        """*mapreduce(ReactiveMP.print_rule_rows, *, mtds)
    )
end

print_rules_table()
```

## Examples and tutorials

Tutorials and examples are available in the [RxInfer documentation](https://biaslab.github.io/RxInfer.jl/stable/).

## Table of Contents

```@contents
Pages = [
  "lib/message.md",
  "lib/node.md",
  "lib/math.md",
  "extra/contributing.md"
]
Depth = 2
```

## Index

```@index
```
