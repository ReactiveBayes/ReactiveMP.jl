SHELL = /bin/bash
.DEFAULT_GOAL = help

.PHONY: format check-format

# No `Pkg.update()` in `scripts_init` on purpose: it defeated `scripts/Manifest.toml` by
# re-resolving the formatter to the newest allowed version on every `make format` /
# `make check-format`, so which formatter you got depended only on when you last ran it --
# and CI and contributors could therefore disagree with no code change involved.
# Use `make scripts_update` to bump deliberately.
scripts_init:
	julia --project=scripts/ -e 'using Pkg; Pkg.instantiate(); Pkg.precompile();'

scripts_update: ## Re-resolve scripts/Manifest.toml (bumps Runic within its compat bound)
	julia --project=scripts/ -e 'using Pkg; Pkg.update(); Pkg.precompile();'

format: scripts_init ## Format Julia code
	julia --project=scripts/ scripts/formatter.jl --overwrite

check-format: scripts_init ## Check Julia code formatting (does not modify files)
	julia --project=scripts/ scripts/formatter.jl

.PHONY: docs

doc_init:
	julia --project=docs -e 'using Pkg; Pkg.instantiate();'

docs: doc_init ## Generate the documentation, running its doctests
	julia --startup-file=no --project=docs docs/make.jl

.PHONY: test test-all test-base test-testutils test-standard test-approximations test-delta test-gaussian-coupling

test: ## Run the fast subset (skips `:slow`). test_args="nodes", "tag:engine", "name:MessageMapping" all work; RUN_AQUA=false skips the slow Aqua checks
	julia -e 'import Pkg; Pkg.activate("."); Pkg.test(test_args = split("$(test_args)") .|> string)'

test-all: ## Run everything, including `:slow`. This is what CI will run
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

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)