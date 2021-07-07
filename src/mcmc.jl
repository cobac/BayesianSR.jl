"""
    step(chain::Chain, base_sample::Sample, i::Int, j::Int ; verbose::Bool = false)

Generates a new Sample of `Chain` by performing one MCMC step from `base_sample`.
`i` is the index of the latest sample.
`j` ∈ {1..k} is the index of the tree to be modified.
"""
function step(chain::Chain, base_sample::Sample, i::Int, j::Int; verbose::Bool=false)
    @unpack k, σ²_prior, σ²_a_prior, σ²_b_prior = chain.hyper

    # Initialize new sample
    old_sample = deepcopy(chain.samples[i])
    proposal = deepcopy(base_sample)

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
    if any_linear_operator_proposal
        verbose && println("There are linear operators in the proposal.")
        proposal.σ²[:σ²_a] = σ²_a = rand(σ²_a_prior)
        proposal.σ²[:σ²_b] = σ²_b = rand(σ²_b_prior)
    end 

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
    
    # isnan(R) && error("R is NaN.")

    # Update chain
    α = min(1.0, R)
    if (rand() < α)
        verbose && println("Sample accepted!")
        chain.stats[:accepted] += 1
        if length(flatten(proposal.trees[j])) > 3
            try
                verbose && @show get_executable(proposal.trees[j], chain.grammar)
                simplified_tree = symbolic_to_RuleNode(get_function(proposal.trees[j], chain.grammar), chain.grammar)
                # Safeguard to avoid creating trees without operators
                # Would break tree movements
                if length(flatten(simplified_tree)) > 2
                    # Safeguard to avoid improper samples
                    log_likelihood(proposal, chain.grammar, chain.x, chain.y)
                    proposal.trees[j] = simplified_tree
                else error()
                end 
                verbose && println("Successful simplification: ", get_executable(proposal.trees[j], chain.grammar))
            catch e
                verbose && println("Failed simplification")
                nothing 
            end
        end
        return proposal
    else 
        verbose && println("Sample rejected.")
        return old_sample
    end 
end 

"""
    step(chain::Chain, i::Int, j::Int ; verbose::Bool = false)

Generates a new Sample of `Chain`.
`i` is the index of the latest sample.
`j` ∈ {1..k} is the index of the tree to be modified.
"""
step(chain::Chain, i::Int, j::Int ; verbose::Bool=false) = step(chain, chain.samples[i], i, j, verbose=verbose)

"""
    mcmc!(chain::Chain, n_steps::Int = 100; verbose::Bool = false, progress::Progress = Progress(n_steps))

Samples from the posterior space `n_steps` iterations via MCMC.
"""
function mcmc!(chain::Chain, n_steps::Int=100; no_chains::Int=1, p_interchain_jump::AbstractFloat=0.05,
               verbose::Bool=false, progress::Progress=Progress(n_steps))
    i₀ = length(chain)
    no_chains < 1 && error("Argument no_chains needs to be a positive integer.")
    if no_chains == 1
        resize!(chain.samples, i₀ + n_steps)
        for i in (i₀ + 1):(i₀ + n_steps)
            verbose && println("==Iteration ", i - i₀, "/", n_steps, " at sample ", i, "==")
            j = chain.stats[:lastj] + 1
            j == no_trees(chain) + 1 ? j = 1 : nothing
            chain.stats[:lastj] = j
            verbose && println("MCMC step for tree j = ", j)
            chain.samples[i] = step(chain, i - 1, j, verbose=verbose)
            chain.stats[:last_sample] += 1
            next!(progress)
        end 
        return nothing
    else # Multichain sampling
        p_interchain_jump > 0 || p_interchain_jump < 1 ||
            error("Argument p_interchain_jump must be between 0 and 1.")
        length(chain) > 1 && error("Multichain sampling is only supported for fresh chains currently.")
        verbose && @warn "Verbose is not fully supported with multichain sampling currently."
        # New vector with all chains initialized with different samples
        chains = push!([Chain([new_sample_recursive(chain.grammar, chain.hyper, chain.x, chain.y)],
                              chain.grammar, chain.x, chain.y, deepcopy(chain.stats), chain.hyper)
                        for _ in 1:(no_chains - 1)], chain)
        for achain in chains resize!(achain.samples, 1 + n_steps) end
        for i in 2:(n_steps + 1)
            verbose && println("==Iteration ", i, "/", n_steps, " at sample ", i, "==")
            j = chain.stats[:lastj] + 1
            j == no_trees(chain) + 1 ? j = 1 : nothing
            verbose && println("MCMC step for tree j = ", j)
            for chain_id in eachindex(chains)
                chains[chain_id].stats[:lastj] = j
                if rand() < p_interchain_jump
                    base_chain = sample(deleteat!([1:no_chains;], chain_id))
                    chains[chain_id].samples[i] = step(chains[chain_id],
                                                       # For now this could be just i - 1, but this is more general for multi-threading
                                                       chains[base_chain].samples[chains[base_chain].stats[:last_sample]],
                                                       i - 1,
                                                       j,
                                                       verbose=false)
                else 
                    chains[chain_id].samples[i] = step(chains[chain_id], i - 1, j, verbose=false)
                end 
                chains[chain_id].stats[:last_sample] += 1
            end 
            next!(progress)
        end 
        return chains
    end 
end 

function log_likelihood(sample::Sample, grammar::Grammar, x::Matrix{Float64}, y::Vector{Float64})
    return logpdf(MvNormal(evalmodel(sample, x, grammar), √sample.σ²[:σ²]), y)
end 

# TODO: Better verbose with native Julia debug macros
# TODO: Multichain sampling with non-fresh chains
# TODO: Implement parallelization for multichain sampling
