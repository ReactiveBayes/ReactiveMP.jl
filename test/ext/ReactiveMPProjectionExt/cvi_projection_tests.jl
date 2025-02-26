@testitem "CVI Projection Extension Tests" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase
    using ReactiveMP
    using Distributions
    using Random

    ReactiveMPProjectionExt = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)
    @test !isnothing(ReactiveMPProjectionExt)
    using .ReactiveMPProjectionExt

    @testset "create_density_function" begin
        # Mock functions and data for testing
        pre_samples = [1.0, 2.0, 3.0]
        
        # Mock message with a simple normal distribution
        m_in = NormalMeanVariance(0.0, 1.0)
        
        # Mock logp_nc_drop_index function that just returns a constant + the input value
        logp_nc_drop_index = (z, i, samples) -> -0.5 * z^2
        
        # Test when forms match (should not include the message logpdf)
        forms_match = true
        df_match = ReactiveMPProjectionExt.create_density_function(forms_match, 1, pre_samples, logp_nc_drop_index, m_in)
        @test df_match(0.5) ≈ logp_nc_drop_index(0.5, 1, pre_samples)
        @test df_match(1.0) ≈ -0.5 # Just the logp_nc_drop_index result
        
        # Test when forms don't match (should include the message logpdf)
        forms_match = false
        df_no_match = ReactiveMPProjectionExt.create_density_function(forms_match, 1, pre_samples, logp_nc_drop_index, m_in)
        # Expected: logp_nc_drop_index + logpdf of the message
        expected_value = logp_nc_drop_index(0.5, 1, pre_samples) + logpdf(m_in, 0.5)
        @test df_no_match(0.5) ≈ expected_value
    end
end