SHELL = /bin/bash
.DEFAULT_GOAL = help

.PHONY: help scripts_init scripts_update format check-format

# `scripts_init` only instantiates `scripts/Manifest.toml`, so every `make format` and
# `make check-format` runs the same pinned Runic; `make scripts_update` bumps it deliberately.
scripts_init:
	julia --project=scripts/ -e 'using Pkg; Pkg.instantiate(); Pkg.precompile();'

scripts_update: ## Re-resolve scripts/Manifest.toml (bumps Runic within its compat bound)
	julia --project=scripts/ -e 'using Pkg; Pkg.update(); Pkg.precompile();'

format: scripts_init ## Format Julia code
	julia --project=scripts/ scripts/formatter.jl --overwrite

check-format: scripts_init ## Check Julia code formatting (does not modify files)
	julia --project=scripts/ scripts/formatter.jl

# Every package has its own documentation site. A site links to the sites of the packages it
# depends on through their inventories (`docs/build/objects.inv`), so `docs-all` builds them in
# dependency order and the ReactiveMP site, which links to every package, last.
.PHONY: doc_init docs docs-all docs-approximations docs-base docs-testutils docs-standard docs-delta docs-gaussian-coupling docs-probit docs-gcv docs-softdot docs-autoregressive docs-continuous-transition docs-polya docs-bifm docs-flow docs-discrete-transition

doc_init:
	julia --project=docs -e 'using Pkg; Pkg.instantiate();'

docs: doc_init ## Build ReactiveMP's documentation site, running its doctests (needs the package sites: `make docs-all`)
	julia --startup-file=no --project=docs docs/make.jl

docs-all: docs-approximations docs-base docs-testutils docs-standard docs-delta docs-gaussian-coupling docs-probit docs-gcv docs-softdot docs-autoregressive docs-continuous-transition docs-polya docs-bifm docs-flow docs-discrete-transition docs ## Build every documentation site, in dependency order

docs-approximations: ## Build lib/MessagePassingRulesApproximations's documentation site into its docs/build
	julia --startup-file=no --project=lib/MessagePassingRulesApproximations/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/MessagePassingRulesApproximations/docs lib/MessagePassingRulesApproximations/docs/make.jl

docs-base: ## Build lib/MessagePassingRulesBase's documentation site into its docs/build
	julia --startup-file=no --project=lib/MessagePassingRulesBase/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/MessagePassingRulesBase/docs lib/MessagePassingRulesBase/docs/make.jl

docs-testutils: ## Build lib/MessagePassingRulesTestUtils's documentation site into its docs/build
	julia --startup-file=no --project=lib/MessagePassingRulesTestUtils/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/MessagePassingRulesTestUtils/docs lib/MessagePassingRulesTestUtils/docs/make.jl

docs-standard: ## Build lib/StandardMessagePassingRules's documentation site into its docs/build
	julia --startup-file=no --project=lib/StandardMessagePassingRules/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/StandardMessagePassingRules/docs lib/StandardMessagePassingRules/docs/make.jl

docs-delta: ## Build lib/DeltaMessagePassingRules's documentation site into its docs/build
	julia --startup-file=no --project=lib/DeltaMessagePassingRules/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/DeltaMessagePassingRules/docs lib/DeltaMessagePassingRules/docs/make.jl

docs-gaussian-coupling: ## Build lib/GaussianCouplingMessagePassingRules's documentation site into its docs/build
	julia --startup-file=no --project=lib/GaussianCouplingMessagePassingRules/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/GaussianCouplingMessagePassingRules/docs lib/GaussianCouplingMessagePassingRules/docs/make.jl

docs-probit: ## Build lib/ProbitMessagePassingRules's documentation site into its docs/build
	julia --startup-file=no --project=lib/ProbitMessagePassingRules/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/ProbitMessagePassingRules/docs lib/ProbitMessagePassingRules/docs/make.jl

docs-gcv: ## Build lib/GCVMessagePassingRules's documentation site into its docs/build
	julia --startup-file=no --project=lib/GCVMessagePassingRules/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/GCVMessagePassingRules/docs lib/GCVMessagePassingRules/docs/make.jl

docs-softdot: ## Build lib/SoftDotMessagePassingRules's documentation site into its docs/build
	julia --startup-file=no --project=lib/SoftDotMessagePassingRules/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/SoftDotMessagePassingRules/docs lib/SoftDotMessagePassingRules/docs/make.jl

docs-autoregressive: ## Build lib/AutoregressiveMessagePassingRules's documentation site into its docs/build
	julia --startup-file=no --project=lib/AutoregressiveMessagePassingRules/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/AutoregressiveMessagePassingRules/docs lib/AutoregressiveMessagePassingRules/docs/make.jl

docs-continuous-transition: ## Build lib/ContinuousTransitionMessagePassingRules's documentation site into its docs/build
	julia --startup-file=no --project=lib/ContinuousTransitionMessagePassingRules/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/ContinuousTransitionMessagePassingRules/docs lib/ContinuousTransitionMessagePassingRules/docs/make.jl

docs-polya: ## Build lib/PolyaMessagePassingRules's documentation site into its docs/build
	julia --startup-file=no --project=lib/PolyaMessagePassingRules/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/PolyaMessagePassingRules/docs lib/PolyaMessagePassingRules/docs/make.jl

docs-bifm: ## Build lib/BIFMMessagePassingRules's documentation site into its docs/build
	julia --startup-file=no --project=lib/BIFMMessagePassingRules/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/BIFMMessagePassingRules/docs lib/BIFMMessagePassingRules/docs/make.jl

docs-flow: ## Build lib/FlowMessagePassingRules's documentation site into its docs/build
	julia --startup-file=no --project=lib/FlowMessagePassingRules/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/FlowMessagePassingRules/docs lib/FlowMessagePassingRules/docs/make.jl

docs-discrete-transition: ## Build lib/DiscreteTransitionMessagePassingRules's documentation site into its docs/build
	julia --startup-file=no --project=lib/DiscreteTransitionMessagePassingRules/docs -e 'using Pkg; Pkg.instantiate()'
	julia --startup-file=no --project=lib/DiscreteTransitionMessagePassingRules/docs lib/DiscreteTransitionMessagePassingRules/docs/make.jl

.PHONY: test test-all test-base test-testutils test-standard test-approximations test-delta test-gaussian-coupling test-probit test-gcv test-softdot test-autoregressive test-continuous-transition test-polya test-bifm test-flow test-discrete-transition

test: ## Run the root suite except items tagged `:slow`. test_args="nodes", "tag:engine", "name:MessageMapping" all work; RUN_AQUA=false skips the slow Aqua checks
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test(test_args = split("$(test_args)") .|> string)'

test-all: ## Run the root suite, `:slow` items included, as CI does
	TEST_ALL=true julia -e 'import Pkg; Pkg.activate("."); Pkg.test(test_args = split("$(test_args)") .|> string)'

test-base: ## Test lib/MessagePassingRulesBase. Takes test_args like `test`, e.g. test_args="tag:quality"
	julia --startup-file=no --project=lib/MessagePassingRulesBase -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-testutils: ## Test lib/MessagePassingRulesTestUtils against the local MessagePassingRulesBase ([sources]). Takes test_args like `test`
	julia --startup-file=no --project=lib/MessagePassingRulesTestUtils -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-standard: ## Test lib/StandardMessagePassingRules against the local lib packages ([sources]). Takes test_args like `test`
	julia --startup-file=no --project=lib/StandardMessagePassingRules -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-approximations: ## Test lib/MessagePassingRulesApproximations, which depends on no sibling. Takes test_args like `test`
	julia --startup-file=no --project=lib/MessagePassingRulesApproximations -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-delta: ## Test lib/DeltaMessagePassingRules against the local lib packages ([sources]). Takes test_args like `test`
	julia --startup-file=no --project=lib/DeltaMessagePassingRules -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-gaussian-coupling: ## Test lib/GaussianCouplingMessagePassingRules against the local lib packages ([sources]). Takes test_args like `test`
	julia --startup-file=no --project=lib/GaussianCouplingMessagePassingRules -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-probit: ## Test lib/ProbitMessagePassingRules against the local lib packages ([sources]). Takes test_args like `test`
	julia --startup-file=no --project=lib/ProbitMessagePassingRules -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-gcv: ## Test lib/GCVMessagePassingRules against the local lib packages ([sources]). Takes test_args like `test`
	julia --startup-file=no --project=lib/GCVMessagePassingRules -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-softdot: ## Test lib/SoftDotMessagePassingRules against the local lib packages ([sources]). Takes test_args like `test`
	julia --startup-file=no --project=lib/SoftDotMessagePassingRules -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-autoregressive: ## Test lib/AutoregressiveMessagePassingRules against the local lib packages ([sources]). Takes test_args like `test`
	julia --startup-file=no --project=lib/AutoregressiveMessagePassingRules -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-continuous-transition: ## Test lib/ContinuousTransitionMessagePassingRules against the local lib packages ([sources]). Takes test_args like `test`
	julia --startup-file=no --project=lib/ContinuousTransitionMessagePassingRules -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-polya: ## Test lib/PolyaMessagePassingRules against the local lib packages ([sources]). Takes test_args like `test`
	julia --startup-file=no --project=lib/PolyaMessagePassingRules -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-bifm: ## Test lib/BIFMMessagePassingRules against the local lib packages ([sources]). Takes test_args like `test`
	julia --startup-file=no --project=lib/BIFMMessagePassingRules -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-flow: ## Test lib/FlowMessagePassingRules against the local lib packages ([sources]). Takes test_args like `test`
	julia --startup-file=no --project=lib/FlowMessagePassingRules -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

test-discrete-transition: ## Test lib/DiscreteTransitionMessagePassingRules against the local lib packages ([sources]). Takes test_args like `test`
	julia --startup-file=no --project=lib/DiscreteTransitionMessagePassingRules -e 'import Pkg; Pkg.test(test_args = split("$(test_args)") .|> string)'

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)