"""
    growtree(grammar::Grammar, hyper::Hyperparams, d::Int)

Samples a branch from the prior distribution.
`d` is the depth of the branch in a tree.
The root node is at `d = 1`.
"""
function growtree(grammar::Grammar, hyper::Hyperparams, d::Int)
    node_types = nodetypes(grammar)
    !in(0, node_types) &&
        error("You are trying to grow a tree from a Grammar without terminal nodes.")
    # Hyper: α, β = 2, 1
    # Prior: Uniform for operators and features
    p₁ = 2 / (1 + d)
    if p₁ > rand()
        node = new_operator(grammar, hyper)
        for child in 1:node_types[node.ind]
            push!(node.children, growtree(grammar, hyper, d + 1))
        end 
    else
        # Node = terminal
        node = new_terminal(grammar)
    end 
    return node
end 

"""
    growtree(grammar::Grammar, hyper::Hyperparams) = growtree(grammar, hyper, d = 1)   

Samples a new tree.
"""
growtree(grammar::Grammar, hyper::Hyperparams) = growtree(grammar, hyper, 1)

"""
    new_operator(grammar::Grammar, hyper::Hyperparams)

Samples a new operator node without children.
If the node is a linear operator, the first child are the linear coefficients.
"""
function new_operator(grammar::Grammar, hyper::Hyperparams)
    node = RuleNode(sample(operator_indices(grammar)))
    # Linear operator
    if node.ind == 2
        θ = RuleNode(1, LinearCoef(hyper))
        push!(node.children, θ)
    end 
    return node
end 

"""
    new_terminal(grammar::Grammar)

Samples a new terminal node.
"""
function new_terminal(grammar::Grammar)
    node = RuleNode(sample(terminal_indices(grammar)))
    return node
end 
