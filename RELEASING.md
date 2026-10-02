# Releasing a package from `lib/`

A working document, like `PLAN.md` and `PHASES.md`, and removed with them at the release. It is
the procedure that took MessagePassingRulesBase from `lib/MessagePassingRulesBase` to its own
repository and registration, on 2026-10-02, written to be repeated for every package under `lib/`
once the packages' grouping is decided (which nodes are standard: user, with Mykola).

Each step says who does it: **agent** (Claude Code with `gh` authenticated as an organisation
admin, `repo` scope), **human** (a decision, or an action only a person can take), or **automatic**
(a bot, once set up).

## The order

A package registers after every sibling it depends on, test-only ones included, since its CI
installs them from the registry:

1. MessagePassingRulesBase and MessagePassingRulesApproximations (no sibling);
2. MessagePassingRulesTestUtils (Base), then StandardMessagePassingRules (Base; TestUtils for its
   tests);
3. every node package (Base, and Approximations or Standard where it says so; TestUtils for its
   tests): Delta, Probit, Pólya, Flow (Approximations), GCV (Approximations and Standard), and
   AR, BIFM, ContinuousTransition, DiscreteTransition, GaussianCoupling, SoftDot.

A new package waits 3 days in General before AutoMerge merges it; a new version of a registered
one, about 15 minutes. Packages with no unregistered dependency can be registered in parallel.

## 1. Decide (human)

- The Julia versions supported: 1.10 and later, for every package (user, 2026-10-02).
- The repository name, `ReactiveBayes/<Package>.jl`: AutoMerge requires the URL to end in
  `/<Package>.jl.git`, and the name to be at least five characters, start upper-case and include
  neither `julia` nor a `Ju` prefix nor a `jl` suffix.
- Whether an existing repository of that name is reused. MessagePassingRulesBase's existed, an
  abandoned earlier attempt never registered, with a different UUID; it was reused and its history
  replaced (user).

## 2. Check that the package stands alone (agent)

- Every dependency is registered, or registers first (the order above); `[sources]` entries for
  siblings go once those are registered.
- `[compat]` has an upper-bounded entry for `julia` and every dependency, `[extras]` included, as
  Aqua's `deps_compat` and AutoMerge both require.
- The suite passes on every Julia version supported, with `TEST_ALL=true`, and on the oldest and
  newest with `Pkg.test(coverage = true)` too, since CI collects coverage. Test copies of the
  package in the scratchpad (`git ls-files -co --exclude-standard`), with `julia +1.10` and so on
  (juliaup), so the package's own Manifest is not rewritten.
- Found on MessagePassingRulesBase, and to check on every package:
  - stdlib compat entries (`LinearAlgebra`, `Pkg`, `Test`) must allow the oldest Julia;
  - an allocation gate that measures through `measure(f, args...)` reports 16 bytes on 1.10 and
    1.11, which do not specialise a function called through a splatted vararg:
    `measure(f::F, args::Vararg{Any, N}) where {F, N}` measures the call itself;
  - JET 0.9, the last for 1.10 and 1.11, fails inside its own report building: `@test_opt` runs
    on 1.12 and later.
- Checked on 1.10 for all packages (2026-10-02, every sibling developed into one environment,
  since 1.10 ignores `[sources]`): every suite, RxInfer, the examples and the course pass, after
  two tests that used `Base.ispublic` (1.11 and later) dropped it.
- Changes the package needs land in the monorepo first, with a CHANGELOG entry, so the split
  history carries them.

## 3. Split the history (agent)

```bash
cd ReactiveMP.jl
git subtree split --prefix=lib/<Package> HEAD -b split/<package>
# the split tree must equal the monorepo's directory
[ "$(git rev-parse HEAD:lib/<Package>)" = "$(git rev-parse split/<package>^{tree})" ]
cd .. && git clone --branch split/<package> ReactiveMP.jl <Package>.jl && cd <Package>.jl
git checkout -b main && git branch -D split/<package> && git remote remove origin
git tag -l | xargs git tag -d            # the clone carries ReactiveMP's v1.0.0 … v6.6.0 tags
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
    --msg-filter "grep -v -E 'claude\.ai/(code/)?(session|chat|share)|Claude-Session:'" main
git update-ref -d refs/original/refs/heads/main && git reflog expire --expire=now --all && git gc --prune=now
git log --format=%B | grep -c claude.ai   # must be 0
git -C ../ReactiveMP.jl branch -D split/<package>
git remote add origin git@github.com:ReactiveBayes/<Package>.jl.git
```

- **The inherited tags must go before anything is pushed**: a `v1.0.0` from ReactiveMP would pose
  as the package's release and confuse TagBot.
- Session links in commit messages are stripped (the no-session-links rule); MessagePassingRulesBase
  had 4 of 75 commits with one.

## 4. Make it a repository of its own (agent)

On `main`, in one commit:

- `version = "1.0.0"` in `Project.toml` (AutoMerge refuses a prerelease suffix such as `-DEV`);
- `.gitignore`: `Manifest.toml`, `Manifest-v*.toml`, `docs/build/`, coverage files, `lcov.info`;
- `README.md`: badges (CI, docs stable and dev, Codecov, Runic), `Pkg.add`, and how to test and
  build the docs without the monorepo's `make`;
- `CHANGELOG.md`, Keep a Changelog, with `[1.0.0]` saying where the history comes from;
- `docs/make.jl`: `deploydocs(repo = "github.com/ReactiveBayes/<Package>.jl.git", devbranch =
  "main", push_preview = true)`; the `sibling(...)` links point at the published `objects.inv` of
  each sibling already released, not at `../../<Sibling>/docs/build`;
- `.github/workflows/`, as MessagePassingRulesBase.jl has them:
  - `CI.yml`: tests on 1.10, 1.11, 1.12, `1` and `pre` (`pre` not gating), `TEST_ALL=true`,
    coverage uploaded with `codecov/codecov-action@v7` and `use_oidc: true` (job permission
    `id-token: write`, no token), and a docs job with `julia-docdeploy@v1`, `GITHUB_TOKEN` and
    `DOCUMENTER_KEY`;
  - `Format.yml`: Runic through `fredrikekre/runic-action@v1`;
  - `TagBot.yml` with `ssh: ${{ secrets.DOCUMENTER_KEY }}`, so a tag's docs build;
  - `CompatHelper.yml`, `subdirs = ["", "test", "docs"]`, with `COMPATHELPER_PRIV` the
    Documenter key, so its pull requests run CI.

Then, locally: the Runic check (`julia --project=<ReactiveMP.jl>/scripts -e 'using Runic;
exit(Runic.main(["--check", "."]))'`), the suite, and the docs (`julia --project=docs -e 'import
Pkg; Pkg.instantiate()'`, then `docs/make.jl`; Documenter needs the `origin` remote set to find
the repository).

## 5. Publish (agent, after the human's go-ahead)

- Create the repository, or, when reusing one, close its open pull requests with a comment
  (`gh pr close N --delete-branch --comment …`), delete its other branches and its old
  `gh-pages`, and force-push `main` only:
  `gh repo create ReactiveBayes/<Package>.jl --public` / `git push --force -u origin main`.
- The Documenter key, never printed and deleted afterwards:

  ```bash
  ssh-keygen -q -t ed25519 -N "" -C Documenter -f $K/key
  gh repo deploy-key add $K/key.pub -R ReactiveBayes/<Package>.jl --title Documenter --allow-write
  base64 -i $K/key | tr -d '\n' | gh secret set DOCUMENTER_KEY -R ReactiveBayes/<Package>.jl
  rm -rf $K
  ```

- The homepage: `gh repo edit --homepage https://reactivebayes.github.io/<Package>.jl/stable/`.

Nothing to install: the Registrator app (`juliateam-registrator`) and the Codecov app are
installed on every ReactiveBayes repository.

## 6. Check it works (agent)

- CI green on every version, Format green (`gh run watch <id> --exit-status`).
- Coverage reached Codecov: the upload step's log says "Upload queued for processing complete",
  and `https://api.codecov.io/api/v2/github/ReactiveBayes/repos/<Package>.jl/` shows
  `"activated": true` and, once processed, its totals.
- GitHub Pages switches itself on at Documenter's first push to `gh-pages`
  (`gh api repos/ReactiveBayes/<Package>.jl/pages/builds/latest`), and
  `https://reactivebayes.github.io/<Package>.jl/dev/` answers 200; `stable/` only after the tag.

## 7. Register (agent posts as the human; the human approves)

```bash
gh api repos/ReactiveBayes/<Package>.jl/commits/<sha>/comments -f body="@JuliaRegistrator register

Release notes:

…"
```

- Only collaborators and **public** organisation members can trigger Registrator: the account
  posting must be one.
- Registrator answers on the commit with the General pull request it opened. AutoMerge checks it;
  a failing check is reported to the human, not fixed on the spot.
- A new package merges after 3 days, if no one objects; then, automatically, TagBot tags `v1.0.0`
  and makes a GitHub release, and the docs job builds `stable/` from the tag.
- **Any comment on the General pull request pauses its auto-merge**, unless it contains
  `[noblock]`. Leave the pull request alone, or write `[noblock]` in every comment.

## 8. The monorepo (agent)

- From the push on, the new repository is the source of truth: the monorepo's copy is frozen, not
  edited (`CLAUDE.md`).
- Once registered: the siblings, ReactiveMP, RxInfer's branch and the docs environments depend on
  the registered version (their `[sources]` path entries for it go), `lib/<Package>` is deleted,
  and the Makefile targets, `LibTests.yml` and `docs-all` lose it; sibling sites link to its
  published `objects.inv`.

## Log

- **MessagePassingRulesBase 1.0.0** (2026-10-02): 1.10 support committed in the monorepo
  (`7609478dc`); repository reused, its 6 bot pull requests closed and old `gh-pages` deleted;
  history of 76 commits plus the standalone commit `6194f45`; CI green on 1.10, 1.11, 1.12, 1.13
  and the prerelease, docs live at `/dev/`, coverage 96.64% (1 815 of 1 878 lines); registration
  [JuliaRegistries/General#170326](https://github.com/JuliaRegistries/General/pull/170326), every
  AutoMerge guideline met, merging after the 3-day wait.
