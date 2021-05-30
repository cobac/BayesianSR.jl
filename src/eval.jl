
"""
    evaltree(tree::RuleNode, x, grammar::Grammar)

Evaluates an `RuleNode`. Outputs a vector with a value for every observation.
"""
function evaltree(tree::RuleNode, x::Matrix, grammar::Grammar, table::SymbolTable)
    n = size(x, 1)
    out = Vector{Float64}(undef, n)
    eq = get_executable(tree, grammar)
    @inbounds for i in 1:n
        tableforeval!(table, x, i) 
        out[i] = Core.eval(table, eq)
    end 
    return out
end 

"""
    evalsample(sample::Sample, x, grammar::Grammar)

Evaluates a `Sample`. Outputs a matrix with the evaluations of every `RuleNode`.
"""
function evalsample(sample::Sample, x::Matrix, grammar::Grammar)
    out = Matrix{Float64}(undef, size(x, 1), length(sample.trees))
    table = SymbolTable(grammar, BayesianSR)
    @inbounds for j in eachindex(sample.trees)
        out[:, j] = evaltree(sample.trees[j], x, grammar, table)
    end 
    return out
end 

"""
    tableforeval!(table::ExprRules.SymbolTable, x, i)

Updates a symbol table to evaluate the i-th observation in a `RuleNode`.
"""
function tableforeval!(table::SymbolTable, x::Matrix, i::Int)
    @inbounds for m in axes(x, 2)
        table[feature_symbols[m]] = x[i, m]
    end 
    return nothing
end 

"""
    evalmodel(sample::Sample, x::Matrix, grammar::Grammar)

Evaluates a `Sample`. Outputs a vector.
"""
function evalmodel(sample::Sample, x::Matrix, grammar::Grammar)
    return sample.β[begin] .+ evalsample(sample, x, grammar) * view(sample.β, 2:length(sample.β))
end 

"""
    evalmodel(chain::Chain)

Evaluates the last `Sample` of a `Chain`.
"""
evalmodel(chain::Chain) = evalmodel(chain.samples[end], chain.x, chain.grammar)

"""
    evalmodel(chain::Chain, x::Matrix)

Evaluates the last `Sample` of a `Chain` with data `x`.
"""
evalmodel(chain::Chain, x::Matrix) = evalmodel(chain.samples[end], x, chain.grammar)

const feature_symbols = [:x1,   :x2,  :x3,  :x4,  :x5,  :x6,  :x7,  :x8,  :x9, :x10,
                         :x11, :x12, :x13, :x14, :x15, :x16, :x17, :x18, :x19, :x20,
                         :x21, :x22, :x23, :x24, :x25, :x26, :x27, :x28, :x29, :x30,
                         :x31, :x32, :x33, :x34, :x35, :x36, :x37, :x38, :x39, :x40]
