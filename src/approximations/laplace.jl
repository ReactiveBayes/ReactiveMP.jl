export LaplaceApproximation, laplace

laplace() = LaplaceApproximation()

struct LaplaceApproximation <: AbstractApproximationMethod end

approximation_name(::LaplaceApproximation)       = "LaplaceApproximation"
approximation_short_name(::LaplaceApproximation) = "LP"

using ForwardDiff
using Optim

#TODO
function gradientOptimization(log_joint::Function, d_log_joint::Function, m_initial, step_size, callback = nothing)
    dim_tot = length(m_initial)
    m_total = zeros(dim_tot)
    m_average = zeros(dim_tot)
    m_new = zeros(dim_tot)
    m_old = m_initial
    satisfied = false
    step_count = 0

    while !satisfied
        m_new = m_old .+ step_size.*d_log_joint(m_old)
        if callback !== nothing
            callback(m_new)
        end
        if log_joint(m_new) > log_joint(m_old)
            proposal_step_size = 10*step_size
            m_proposal = m_old .+ proposal_step_size.*d_log_joint(m_old)
            if log_joint(m_proposal) > log_joint(m_new)
                m_new = m_proposal
                step_size = proposal_step_size
            end
        else
            step_size = 0.1*step_size
            m_new = m_old .+ step_size.*d_log_joint(m_old)
        end
        step_count += 1
        m_total .+= m_old
        m_average = m_total ./ step_count
        if step_count > 5
            if sum(sqrt.(((m_new.-m_average)./m_average).^2)) < dim_tot*0.01
                satisfied = true
            elseif abs(log_joint(m_new) - log_joint(m_old)) < 1e-4
                satisfied = true
            end
        end
        if step_count > dim_tot*250
            satisfied = true
        end
        m_old = m_new
    end

    return m_new
end

function approximate_meancov(::LaplaceApproximation, g::Function, distribution)
    
    logg = (z) -> log(g(z))
    logd = (z) -> logpdf(distribution, z)

    logf   = (z) -> logg(z) + logd(z)
    d_logf = (z) -> ForwardDiff.gradient(logf, z)

    m = gradientOptimization(logf, d_logf, mean(distribution), 0.01)

    # result = optimize((d) -> -(logf(d)), mean(distribution), LBFGS())
    # if !Optim.converged(result)
    #     @show result
    #     throw("LaplaceApproximation: convergence failed")
    # end

    # m = Optim.minimizer(result)
    c = -inv(Matrix(Hermitian(ForwardDiff.hessian(logf, m))))

    return m ,c
end

function approximate_kernel_expectation(::LaplaceApproximation, g::Function, distribution)
    return approximate_kernel_expectation(srcubature(), g, distribution)
end