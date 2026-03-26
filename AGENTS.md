# Running tests
You can run specific test cases filtered by name using TestEnv (to activate the test pkg environment) and TestItemRunner (to run the tests):

```julia
julia --pkgimages=existing --project -e "
using TestEnv
TestEnv.activate()

using TestItemRunner
@run_package_tests filter=ti -> occursin(\"TEST NAME\", ti.name)
"
```