"""
    Hyperparams(k = 3::Int, σ²_prior = InverseGamma(0.5, 0.5)::UnivariateDistribution)

Hyperparameters of a `Chain`.
"""
@with_kw struct Hyperparams
    k = 3::Int
    σ²_prior::UnivariateDistribution = InverseGamma(0.5, 0.5)
    σ²_a_prior::UnivariateDistribution = InverseGamma(0.5, 0.5)
    σ²_b_prior::UnivariateDistribution = InverseGamma(0.5, 0.5)
end 
