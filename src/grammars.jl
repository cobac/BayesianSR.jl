"""
`Grammar` with default operators.
"""
const defaultgrammar = @grammar begin
    Real = Real + Real
    Real = Real - Real
    Real = Real * Real 
    Real = Real / Real
    Real = cos(Real) 
    Real = sin(Real) 
end

"""
`Grammar` with the components of the linear operator `lt()`.
"""
const lineargrammar = @grammar begin
    # Expressions get evaluated at global scope, need to export or include module
    LinearCoef = _(BayesianSR.LinearCoef(0, 0)) # Always ind = 1
    Real = linear_operator(LinearCoef, Real) # Always ind = 2
end 

"""
    lt(lc::LinearCoef, x)

Linear operator.

See also: `LinearCoef`
"""
function linear_operator(lc::LinearCoef, x::Real)
    lc.a + lc.b * x
end

"""
    variablestogrammar(x)

Creates a `Grammar` with all the features in `x`.
"""
function variablestogrammar(x)
    k = size(x, 2)
    rules = [Symbol("x", i) for i in 1:k]
    types = [Symbol("Real") for _ in 1:k]
    isterminal = ones(Int, k)
    iseval = zeros(Int, k)
    childtypes = [Symbol[] for _ in 1:k]
    bytype = Dict(:Real => [i for i in 1:k])
    
    grammar = Grammar(rules, types, isterminal, iseval, bytype, childtypes)
    return grammar
end 

"""
    raw_nodetypes(grammar::Grammar)

Returns a vector with the types of possible nodes from a `Grammar`.
- 1: unary operator
- 2: binary operator
- 0: terminal node
"""
function raw_nodetypes(grammar::Grammar)
    types = [nchildren(grammar, i)
             for i in 1:length(grammar)]
    return types
end 

"""
    nodetypes(grammar::Grammar)

Returns a vector with the types of possible nodes from a BayesianSR `Grammar`.
- 1: unary operator
- 2: binary operator
- 0: terminal node
- -1: coefficients of the linear operator
"""
function nodetypes(grammar::Grammar)
    types = raw_nodetypes(grammar)
    # LinearCoef is not considered a node
    types[1] = -1
    # Linear operator is unary
    types[2] = 1
    return types
end 

"""
    operator_indices(grammar::Grammar)

Returns a vector with the indices of all operators in a grammar.
"""
function operator_indices(grammar::Grammar)
    node_types = nodetypes(grammar)
    is = findall(x -> x==1 || x==2, node_types)
    return is
end 

"""
    terminal_indices(grammar::Grammar)

Returns a vector with the indices of all terminals in a grammar.
"""
function terminal_indices(grammar::Grammar)
    node_types = nodetypes(grammar)
    is = findall(iszero, node_types)
    return is
end 
