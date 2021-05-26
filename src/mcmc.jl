"""
    step(chain::Chain, i::Int, j::Int ; verbose::Bool = false)

Generates a new Sample of `Chain`.
`i` is the index of the latest sample.
`j` ∈ {1..k} is the index of the tree to be modified.
"""
function step(chain::Chain, i::Int, j::Int ; verbose::Bool=false)
    @unpack k, σ²_prior, σ²_a_prior, σ²_b_prior = chain.hyper

    # Initialize new sample
    old_sample = deepcopy(chain.samples[i])
    proposal = deepcopy(chain.samples[i])

    # Maybe gather old LinearCoef parameters
    any_linear_operator_old = any_linear_operators(old_sample.trees[j])
    if any_linear_operator_old
        verbose && println("There are linear operators in the old sample.")
        θ_old = recover_LinearCoef(old_sample.trees[j])
        # Get last LinearCoef variances for this (j-th) tree
        # First k samples variances are not real
        i > k ? previous_i = i - k : previous_i = i
        old_σ²_a = chain.samples[previous_i].σ²[:σ²_a]
        old_σ²_b = chain.samples[previous_i].σ²[:σ²_b]
    end 

    # Propose a new tree
    proposal_tree = proposetree(proposal.trees[j], chain.grammar, chain.hyper,
                                verbose=verbose)

    # Check for linear operators in proposal
    n_linear_operators_proposal = n_linear_operators(proposal_tree.tree)
    any_linear_operator_proposal = n_linear_operators_proposal > 0
    verbose && any_linear_operator_proposal &&
        println("There are linear operators in the proposal.")
    # Even if there are no linear operators,
    # we might need to sample variances to calculate rjmcmc_h⁺ in the shrinkage edge case
    proposal.σ²[:σ²_a] = σ²_a = rand(σ²_a_prior)
    proposal.σ²[:σ²_b] = σ²_b = rand(σ²_b_prior)

    # Generate new LinearCoefs and auxiliary variables U
    if any_linear_operator_old && any_linear_operator_proposal
        if length(θ_old.a) < n_linear_operators_proposal
            rjmcmc = :expansion
            verbose && println("Reversible Jump MCMC: ", rjmcmc)
            θ_proposal, U, U⁺ = propose_LinearCoef!(proposal_tree.tree, σ²_a, σ²_b, θ_old, rjmcmc)
            rjmcmc_h = sum(logpdf.(Normal(1, σ²_a), U.a)) +
                sum(logpdf.(Normal(0, σ²_b), U.b))
            rjmcmc_h⁺ = sum(logpdf.(Normal(1, old_σ²_a), U⁺.a)) +
                sum(logpdf.(Normal(0, old_σ²_b), U⁺.b))
            jacobian = log(2.0^(-2 * length(θ_old.a)))
        elseif length(θ_old.a) > n_linear_operators_proposal
            rjmcmc = :shrinkage
            verbose && println("Reversible Jump MCMC: ", rjmcmc)
            θ_proposal, U, U⁺ = propose_LinearCoef!(proposal_tree.tree, σ²_a, σ²_b, θ_old, rjmcmc)
            rjmcmc_h = sum(logpdf.(Normal(0, σ²_a), U.a)) +
                sum(logpdf.(Normal(0, σ²_b), U.b))
            rjmcmc_h⁺ = sum(logpdf.(Normal(0, old_σ²_a), U⁺.a)) +
                sum(logpdf.(Normal(0, old_σ²_b), U⁺.b))
            jacobian = log(2.0^(2 * length(θ_proposal.a)))
        else # Same dimensions
            θ_proposal = propose_LinearCoef!(proposal_tree.tree, σ²_a, σ²_b)
        end 
        # Edge cases: 
    elseif any_linear_operator_proposal # From 0 to some linear operator
        rjmcmc = :expansion
        verbose && println("Edge case: expanding from 0 linear operators.")
        θ_proposal = propose_LinearCoef!(proposal_tree.tree, σ²_a, σ²_b)
        # θ_proposal = U
        rjmcmc_h = sum(logpdf.(Normal(1, σ²_a), θ_proposal.a)) +
            sum(logpdf.(Normal(0, σ²_b), θ_proposal.b))
        rjmcmc_h⁺ = 0
        jacobian = 0
    elseif any_linear_operator_old # From some linear operator to 0
        rjmcmc = :shrinkage
        verbose && println("Edge case: shrinking to 0 linear operators.")
        # U⁺ = θ_old
        rjmcmc_h = 0
        rjmcmc_h⁺ = sum(logpdf.(Normal(1, old_σ²_a), θ_old.a)) +
            sum(logpdf.(Normal(0, old_σ²_b), θ_old.b))
        jacobian = 0
    end 

    # Update the proposal
    proposal.trees[j] = proposal_tree.tree
    try 
        optimβ!(proposal, chain.x, chain.y, chain.grammar)
    catch e 
        verbose && println("Rejected proposal with an improper sample: ", proposal_tree.tree)
        return old_sample
    end 
    proposal.σ²[:σ²] = rand(σ²_prior)

    # Calculate R
    numerator = log_likelihood(proposal, chain.grammar, chain.x, chain.y) + # Likelihood
        logpdf(σ²_prior, proposal.σ²[:σ²]) + # σ² prior
        # Trees prior
        sum([tree_p(tree, chain.grammar) for tree in proposal.trees]) +
        # Probability of reverse tree jump (proposal -> old)
        proposal_tree.p_mov_inv

    if any_linear_operator_proposal
        numerator += sum(logpdf.(Normal(1, √σ²_a), θ_proposal.a)) + # P intercepts
            sum(logpdf.(Normal(1, √σ²_b), θ_proposal.b)) + # P slopes
            logpdf(σ²_a_prior, σ²_a) + # σ²_a prior
            logpdf(σ²_b_prior, σ²_b)  # σ²_b prior
    end 

    denominator = log_likelihood(old_sample, chain.grammar, chain.x, chain.y) + # Likelihood
        logpdf(σ²_prior, old_sample.σ²[:σ²]) + # σ² prior 
        # Trees prior
        sum([tree_p(tree, chain.grammar) for tree in old_sample.trees]) +
        # Probability of tree jump (old -> proposal)
        proposal_tree.p_mov

    if any_linear_operator_old
        denominator += sum(logpdf.(Normal(1, √old_σ²_a), θ_old.a)) + # P intercepts
            sum(logpdf.(Normal(1, √old_σ²_b), θ_old.b)) + # P slopes
            logpdf(σ²_a_prior, old_σ²_a) + # σ²_a prior
            logpdf(σ²_b_prior, old_σ²_b)  # σ²_b prior
    end 


    # Reversible Jump MCMC required
    if @isdefined rjmcmc
        numerator += rjmcmc_h⁺ + jacobian
        denominator += rjmcmc_h
    end 
    
    R = exp(numerator - denominator)
    
    if verbose
        println("Acceptance ratio log-numerator: ", numerator)
        println("Acceptance ratio log-denominator: ", denominator)
        println("Acceptance ratio: ", R)
    end 
    
    isnan(R) && error("R is NaN.")

    # Update chain
    α = min(1.0, R)
    if (rand() < α)
        verbose && println("Sample accepted!")
        chain.stats[:accepted] += 1
        return proposal
    else 
        verbose && println("Sample rejected.")
        return old_sample
    end 
end 

"""
    mcmc!(chain::Chain, n_steps::Int = 100; verbose::Bool = false)

Samples from the posterior space `n_steps` iterations via MCMC.
"""
function mcmc!(chain::Chain, n_steps::Int=100; verbose::Bool=false)
    i₀ = length(chain)
    resize!(chain.samples, i₀ + n_steps)
    for i in (i₀ + 1):(i₀ + n_steps)
        verbose && println("==Iteration ", i-i₀, "/", n_steps, " at sample ", i, "==")
        j = chain.stats[:lastj] + 1
        j == no_trees(chain) + 1 ? j = 1 : nothing
        chain.stats[:lastj] = j
        verbose && println("MCMC step for tree j = ", j)
        chain.samples[i] = BayesianSR.step(chain, i - 1, j, verbose=verbose)
    end 
end 

function log_likelihood(sample::Sample, grammar::Grammar, x::Matrix{Float64}, y::Vector{Float64})
    logpdf(MvNormal(sample.β[begin] .+
        evalsample(sample, x, grammar) * view(sample.β, 2:length(sample.β)),
                    √sample.σ²[:σ²]),
           y)
end 
