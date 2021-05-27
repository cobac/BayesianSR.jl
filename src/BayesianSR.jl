module BayesianSR
using ExprRules, Distributions, Random, StatsBase, Parameters, Reexport
import ExprRules: RuleNodeAndCount, RuleNode

export Chain, Hyperparams, mcmc!, no_trees
@reexport using ExprRules: @grammar

# Hyperparameters of the model
include("hyperparams.jl")
# Each sample of the MCMC chain
include("sample.jl")
# A MCMC chain
include("chain.jl")
# Coefficients for linear operators
include("linearcoef.jl")
# Grammar of possible operators
include("grammars.jl")
# Functions to evaluate symbolic trees
include("eval.jl")
# Optimize symbolic tree weights via OLS
include("ols.jl")
# Utility functions for symbolic trees
include("utils.jl")
# Grow new symbolic trees
include("growtree.jl")
# Calculate the prior probability of symbolic trees
include("treeprior.jl")
# Possible movements to alter symbolic trees
include("treemovements.jl")
# Symbolic tree structure MCMC proposal 
include("treeproposal.jl")
# Linear coefficients MCMC proposal
include("coefproposal.jl")
# Explore posterior space
include("mcmc.jl")

"""
    RuleNode(grammar::Grammar, hyper::Hyperparams)

Samples a random `RuleNode` from the tree prior distribution with a `Grammar`.

See also: `growtree`
"""
ExprRules.RuleNode(grammar::Grammar, hyper::Hyperparams) = growtree(grammar, hyper)

end # module
