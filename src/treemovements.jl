"""
    ChangedTree(tree::RuleNode, changed_node::RuleNode, d::Int)

Type with the information required to calculate the transition probabilities of growing or pruning a tree.

- `tree`: The new tree
- `changed_node`: The new branch or the pruned branch.
- `d`: The depth of `changed_node`.

See also: `grow!`, `prune!`
"""
struct ChangedTree
    tree::RuleNode
    changed_node::RuleNode
    d::Int
end 

"""
    grow!(node::RuleNode, grammar::Grammar)

Tree movement grow.
Selects a random terminal node and replaces it with a new branch from the tree prior.

See also: `growtree`
"""
function grow!(node::RuleNode, grammar::Grammar, hyper::Hyperparams)
    loc = sampleterminal(node, grammar)
    old_node = get(node, loc)
    d = node_depth(node, old_node)
    old_node = deepcopy(old_node)
    new_node = growtree(grammar, hyper, d)
    insert!(node, loc, new_node)
    return ChangedTree(node, new_node, d)
end 

"""
    prune!(node::RuleNode, grammar::Grammar)

Tree movement prune.
Selects a random operator node and replaces it with a terminal node.
"""
function prune!(node::RuleNode, grammar::Grammar)
    loc = sampleoperator(node, grammar)
    old_node = get(node, loc)
    d = node_depth(node, old_node)
    old_node = deepcopy(old_node)
    insert!(node, loc, new_terminal(grammar))
    return ChangedTree(node, old_node, d)
end 

"""
    DeletedTree(tree::RuleNode,
    changed_node::Union{Nothing, RuleNode},
    d::Int,
    p_child::Float64)

Type with the information required to calculate the transition probabilities of deleting a node.

- `tree`: Is the new tree.
- `dropped_node`: Maybe a dropped branch. If we delete a binary operator we only keep one of its children.
- `d`: Depth of the deleted node.
- `p_child`: Probability of selecting the child that replaces the parent node.

See also: `delete!`
"""
struct DeletedTree
    tree::RuleNode
    dropped_node::Union{Nothing, RuleNode}
    d::Int 
    p_child::Float64
end 

"""
    delete!(node::RuleNode, grammar::Grammar)

Tree movement delete.
Selects a random candidate for deletion node and replaces it with one of its children.

See also: `iscandidate`
"""
function delete!(node::RuleNode, grammar::Grammar)
    loc = samplecandidate(node, grammar)
    target = get(node, loc)
    d = node_depth(node, target)
    n_children = length(target.children)
    if target.ind == 2
        p_child = 1
        dropped_node = nothing
        insert!(node, loc, target.children[2])
    elseif n_children == 1
        p_child = 1
        dropped_node = nothing
        insert!(node, loc, target.children[1])
    elseif target != node # no unary and no root = binary no root
        p_child = 0.5
        i = sample(1:2)
        dropped_node = target.children[i == 1 ? 2 : 1]
        insert!(node, loc, target.children[i])
    else # only binary root left
        operator_is = operator_indices(grammar)
        ind_children = [child.ind for child in target.children]
        ind_operator = findall(x -> in(x, operator_is), ind_children)
        p_child = 1/length(ind_operator)
        i = sample(ind_operator)
        dropped_node = target.children[i == 1 ? 2 : 1]
        node = target.children[i]
    end 
    return DeletedTree(node, dropped_node, d, p_child)
end 

"""
    InsertedTree(tree::RuleNode,
    new_branch::Union{Nothing, RuleNode},
    d::Int)

Type with the information required to calculate the transition probabilities of inserting a node.

- `tree`: Is the new tree.
- `new_branch`: Maybe a new branch.
- `d`: Depth of the deleted node.

See also: `insert_node!`
"""
struct InsertedTree
    tree::RuleNode
    new_branch::Union{Nothing, RuleNode}
    d::Int 
end 

"""
    insert_node!(node::RuleNode, grammar::Grammar)

Tree movement insert.
Select a random node and insert a new operator node between it and its parent.
If the new operator is binary we grow the second children from the tree prior.

See also: `growtree`
"""
function insert_node!(node::RuleNode, grammar::Grammar, hyper::Hyperparams)
    node_types = nodetypes(grammar)
    loc = sample(NodeLoc, node)
    old = get(node, loc)
    while_count = 0
    while old.ind == 1 # No LinearCoef node
        while_count += 1
        while_count > 1000 && error("insert_node!() got stuck in a infinite loop.")
        loc = sample(NodeLoc, node)
        old = get(node, loc)
    end 
    new = new_operator(grammar, hyper)
    type = node_types[new.ind]
    d = node_depth(node, old)
    old = deepcopy(old)
    if type == 2
        new_branch = growtree(grammar, hyper, d)
        new.children = [old, new_branch]
    else # type == 1
        push!(new.children, old)
        new_branch = nothing
    end 
    insert!(node, loc, new)
    return InsertedTree(node, new_branch, d)
end 

"""
    ReassignedTree(tree::RuleNode,
    changed_node::Union{Nothing, RuleNode},
    d::Int,
    transition::Symbol)

Type with the information required to calculate the transition probabilities of reassigning an operator node.

- `tree`: Is the new tree.
- `changed_node`: Maybe a new branch, or a dropped branch.
- `d`: Depth of the changed branch.
- `transition`: `:un2bin`, `:bin2un` or `:same`.

See also: `insert_node!`
"""
struct ReassignedTree
    tree::RuleNode
    changed_node::Union{Nothing, RuleNode}
    d::Int
    transition::Symbol
end 

"""
    re_operator!(node::RuleNode, grammar::Grammar)

Tree movement reassign operator.
Selects a random operator node and replaces it with another one.
If the old node was unary and the new node is binary,
we sample its second children from the tree prior.
If the old node was binary and the new node is unary,
we drop its second children.
"""
function re_operator!(node::RuleNode, grammar::Grammar, hyper::Hyperparams)
    node_types = nodetypes(grammar)
    operator_is = operator_indices(grammar)
    loc = sampleoperator(node, grammar)
    target =get(node, loc)
    d = node_depth(node, target)
    target = deepcopy(target)
    old_ind = target.ind
    old_type = node_types[old_ind]
    # Remove old index from the pool of operators
    operator_rm = findfirst(isequal(target.ind), operator_is)
    deleteat!(operator_is, operator_rm)
    # Sample new operator
    new = new_operator(grammar, hyper)
    new_ind = new.ind
    while !in(new_ind, operator_is)
        new = new_operator(grammar, hyper)
        new_ind = new.ind
    end 
    new_type = node_types[new_ind]
    # Reassign the operator
    target.ind = new_ind
    # Cases with linear operators
    if old_ind == 2 
        if new_type == 2 # Linear operator -> Binary
            transition = :un2bin
            changed_node = growtree(grammar, hyper, d)
            popfirst!(target.children) # Remove LinearCoef
            push!(target.children, changed_node)
        else # Linear operator -> Unary
            transition = :same
            changed_node = nothing
            popfirst!(target.children) # Remove LinearCoef
        end 
        node = insert!(node, loc, target)
    elseif new_ind == 2 
        if old_type == 2 # Binary -> Linear operator
            transition = :bin2un
            changed_node = target.children[2]
            deleteat!(target.children, 2)
            pushfirst!(target.children, RuleNode(1, LinearCoef(hyper)))
        else # Unary -> Linear operator
            transition = :same
            changed_node = nothing
            pushfirst!(target.children, RuleNode(1, LinearCoef(hyper))) # Add LinearCoef
        end 
        node = insert!(node, loc, target)
    else # No linear operator involved
        if new_type == 2 && old_type == 1
            transition = :un2bin
            changed_node = growtree(grammar, hyper, d)
            push!(target.children, changed_node)
        elseif new_type == 1 && old_type == 2
            transition = :bin2un
            changed_node = target.children[2]
            deleteat!(target.children, 2)
        else # new_type == old_type
            transition = :same
            changed_node = nothing
        end 
        insert!(node, loc, target)
    end 
    return ReassignedTree(node, changed_node, d, transition)
end 

"""
    re_feature!(node::RuleNode, grammar::Grammar)

Tree movement reassign feature.
Selects a random terminal node and replaces it with another one.
"""
function re_feature!(node::RuleNode, grammar::Grammar)
    terminal_is = terminal_indices(grammar)
    loc = sampleterminal(node, grammar)
    target = get(node, loc)
    # Remove old index from the pool of terminals
    terminal_rm = findfirst(isequal(target.ind), terminal_is)
    deleteat!(terminal_is, terminal_rm)
    # Sample new terminal
    new = new_terminal(grammar)
    while !in(new.ind, terminal_is)
        new = new_terminal(grammar)
    end 
    insert!(node, loc, new)
    return node
end 
