"""
    flatten(node::RuleNode)

Flattens a `RuleNode` as a vector of indices,
excluding nodes containing the coefficients of the linear operators.
"""
function flatten(node::RuleNode)
    out = Int[]
    queue = [node]
    while !isempty(queue)
        push!(out, queue[1].ind)
        append!(queue, queue[1].children)
        deleteat!(queue, 1)
    end
    # Delete LinearCoef nodes
    filter!(i -> i != 1, out)
    return out
end

"""
    length(grammar::Grammar) = length(grammar.rules)

Number of indices in a `Grammar`.
"""
Base.length(grammar::Grammar) = length(grammar.rules)

"""
    sampleterminal(node::RuleNode, grammar::Grammar)

Samples a random terminal node via rejection sampling.
"""
function sampleterminal(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    operator_is = operator_indices(grammar)
    out = sample(NodeLoc, node)
    # if node is LinearCoef or operator, resample
    target = get(node, out)
    while target.ind == 1|| in(target.ind, operator_is)
        out = sample(NodeLoc, node)
        target = get(node, out)
    end 
    return out
end 

"""
    sampleoperator(node::RuleNode, grammar::Grammar)

Samples a random operator node via rejection sampling.
"""
function sampleoperator(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    terminal_is = terminal_indices(grammar)
    out = sample(NodeLoc, node)
    # if node is LinearCoef or terminal, resample
    target = get(node, out)
    while target.ind == 1 || in(target.ind, terminal_is)
        out = sample(NodeLoc, node)
        target = get(node, out)
    end 
    return out
end 

"""
    samplecandidate(node::RuleNode, grammar::Grammar)

Samples a random candidate for deletion node via rejection sampling.

See also: `iscandidate`
"""
function samplecandidate(node::RuleNode, grammar::Grammar)
    out = sample(NodeLoc, node)
    count = 0
    # if node is not candidate,  resample
    target = get(node, out)
    while !iscandidate(target, node, grammar)
        count += 1
        count >= 1000 && error("`samplecandidate()` got stuck in an infinite loop.")
        out = sample(NodeLoc, node)
        target = get(node, out)
    end 
    return out
end 

"""
    iscandidate(target::RuleNode, root::RuleNode, grammar::Grammar)

Checks whether a node is a candidate for deletion.

Candidate nodes are operator nodes.
The root node can only be a candidate for deletion if it has operator children.
"""
function iscandidate(target::RuleNode, root::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    terminal_is = terminal_indices(grammar)
    operator_is = operator_indices(grammar)
    inds = [child.ind for child in target.children]
    if target.ind == 1
        # LinearCoef
        return false
    elseif in(target.ind, terminal_is)
        # terminal node == false
        return false
    elseif target == root && count(i -> in(i, operator_is), inds) == 0
        # root node with all terminal children == false
        return false
    else
        return true
    end 
end 

"""
    n_operators(node::RuleNode, grammar::Grammar)

Number of operator nodes in a `RuleNode`.
"""
function n_operators(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    operator_is = operator_indices(grammar)
    nodes = flatten(node)
    out = count(i -> in(i, operator_is), nodes)
    return out
end

"""
    n_linear_operators(node::RuleNode)

Number of linear operators in a `RuleNode`.
"""
function n_linear_operators(node::RuleNode)
    nodes = flatten(node)
    return count(i -> i==2, nodes)
end 

"""
    n_terminals(node::RuleNode, grammar::Grammar)

Number of terminal nodes in a `RuleNode`.
"""
function n_terminals(node::RuleNode, grammar::Grammar)
    node_types = nodetypes(grammar)
    terminal_is = terminal_indices(grammar)
    nodes = flatten(node)
    out = count(i -> in(i, terminal_is), nodes)
    return out
end

"""
    n_candidates(node::RuleNode, grammar::Grammar)

Number of candidate for deletion nodes in a `RuleNode`.

See also: `iscandidate`
"""
function n_candidates(node::RuleNode, grammar::Grammar)
    n = n_operators(node, grammar)
    node_types = nodetypes(grammar)
    operator_is = operator_indices(grammar)
    inds = [child.ind for child in node.children]
    if count(i -> in(i, operator_is), inds) == 0
        n -= 1
    end 
    return n
end 
