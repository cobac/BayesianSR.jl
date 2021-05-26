"""
    TreeProposal(tree::RuleNode, movement::Symbol, p_mov::Float64, p_mov_inv::Float64)

A proposal generated in one MCMC iteration.

- `tree`: the new proposed tree.
- `movement`: a symbol indicating the movement that was performed.
- `p_mov`: the probability of jumping from the old tree to the proposal.
- `p_mov_inv`: the probability of jumping from the proposal to the old tree.
"""
struct TreeProposal
    tree::RuleNode
    movement::Symbol
    p_mov::Float64
    p_mov_inv::Float64
end 

"""
    proposetree(tree::RuleNode, grammar::Grammar)

Generates a `TreeProposal`.
"""
function proposetree(tree::RuleNode, grammar::Grammar, hyper::Hyperparams; verbose::Bool = false)
    tree = deepcopy(tree)
    nₒ = n_operators(tree, grammar)
    n_lo = n_linear_operators(tree)
    nₜ = n_terminals(tree, grammar)
    n_cand = n_candidates(tree, grammar)
    node_types = nodetypes(grammar)
    operator_is = operator_indices(grammar)
    terminal_is = terminal_indices(grammar)
    # P_0 = P(tree stays the same)
    p_0 = n_lo > 0 ? n_lo/(4*(n_lo + 3)) : 0
    # P_g = P(tree grows)
    p_g = (1 - p_0)/3 * min(1, 8/(nₒ+2))
    # P_p = P(pruning the tree)
    p_p = (1-p_0)/3-p_g
    # P_d = P(deleting a candidate node)
    p_d = (1-p_0)/3 * n_cand/(n_cand+3)
    # P_i = P(insert a new node)
    p_i  = (1-p_0)/3 - p_d
    # P_ro = P(reassign operator)
    # P_rf = P(reassign feature)
    p_ro = p_rf = (1-p_0)/6
    movements = [:stay , :grow, :prune, :delete, :insert, :re_operator, :re_feature]
    weights =   [p_0   ,   p_g,    p_p,     p_d,     p_i,         p_ro,        p_rf]
    mov = sample(movements, Weights(weights))
    verbose && println("Tree movement: ", mov)

    # Hyper: α, β = 2, 1
    # Prior: Uniform for operators and features
    if mov == :stay
        p = p_inv = log(p_0)
    elseif mov == :grow
        changed_tree = grow!(tree, grammar, hyper)
        tree = changed_tree.tree
        p = log(p_g) + # P of movement = grow
            log(1/nₜ) + # P of selecting this terminal node
            # P of growing this specific branch
            tree_p(changed_tree.changed_node, changed_tree.d, grammar)
        new_nₒ = n_operators(tree, grammar)
        new_n_lo = n_linear_operators(tree)
        new_p_0 = new_n_lo > 0 ? new_n_lo/(4*(new_n_lo + 3)) : 0
        new_p_g = (1 - new_p_0)/3 * min(1, 8/(new_nₒ+2))
        new_p_p = (1-new_p_0)/3-new_p_g
        p_inv = log(new_p_p) + # P of movement = prune
            log(1/new_nₒ) + #P of selecting this operator node
            log(1/length(terminal_is)) # P of selecting this terminal
    elseif mov == :prune
        changed_tree = prune!(tree, grammar)
        tree = changed_tree.tree
        p = log(p_p) + # P of movement = prune
            log(1/nₒ) + # P of selecting this operator node
            log(1/length(terminal_is)) # P of selecting this terminal
        new_nₒ = n_operators(tree, grammar)
        new_n_lo = n_linear_operators(tree)
        new_p_0 = new_n_lo > 0 ? new_n_lo/(4*(new_n_lo + 3)) : 0
        new_p_g = (1 - new_p_0)/3 * min(1, 8/(new_nₒ+2))
        p_inv = log(new_p_g) + # P of movement = grow
            log(1/n_terminals(tree, grammar)) + # P of selecting this terminal node
            # P of growing this specific branch
            tree_p(changed_tree.changed_node, changed_tree.d, grammar)
    elseif mov == :delete
        deleted_tree = delete!(tree, grammar)
        tree = deleted_tree.tree
        p = log(p_d) + # P of movement = delete
            log(1/n_cand) + # P of selecting this candidate node
            log(deleted_tree.p_child) # P of selecting this specific child
        new_n_cand = n_candidates(tree, grammar)
        new_n_lo = n_linear_operators(tree)
        new_p_0 = new_n_lo > 0 ? new_n_lo/(4*(new_n_lo + 3)) : 0
        new_p_d = (1-p_0)/3 * new_n_cand/(new_n_cand+3)
        new_p_i = (1-new_p_0)/3 - new_p_d
        p_inv = log(new_p_i) + # P of movement = insert
            log(1/length(flatten(tree))) + # P of selecting this node
            log(1/length(operator_is))  # P of inserting this operator
        if !isnothing(deleted_tree.dropped_node)
            # If we had to grow a new branch, P of growing that branch
            p_inv += tree_p(deleted_tree.dropped_node, deleted_tree.d, grammar)
        end 
    elseif mov == :insert
        inserted_tree = insert_node!(tree, grammar, hyper)
        tree = inserted_tree.tree
        p = log(p_i) + # P of movement = insert
            log(1/length(flatten(tree))) + # P of selecting this node
            log(1/length(operator_is)) # P of inserting this operator
        if !isnothing(inserted_tree.new_branch)
            # If we had to grow a new branch, P of growing that branch
            p += tree_p(inserted_tree.new_branch, inserted_tree.d, grammar)
        end 
        new_n_cand = n_candidates(tree, grammar)
        new_n_lo = n_linear_operators(tree)
        new_p_0 = new_n_lo > 0 ? new_n_lo/(4*(new_n_lo + 3)) : 0
        new_p_d = (1-p_0)/3 * new_n_cand/(new_n_cand+3)
        p_inv = log(new_p_d) + # P of movement = delete
            log(1/new_n_cand) # P of selecting this candidate node
        if !isnothing(inserted_tree.new_branch)
            # If we had to grow a new branch, P of deleting it
            p_inv += log(1/2)
        end 
    elseif mov == :re_operator
        reassigned_tree = re_operator!(tree, grammar, hyper)
        tree = reassigned_tree.tree
        p = p_inv = log(p_ro) + # P of movement = reassign operator
            log(1/nₒ) + # P of selecting this operator node
            log(1/(length(operator_is) - 1)) # P of choosing the new operator
        if reassigned_tree.transition == :un2bin
            # P of growing the new branch
            p += tree_p(reassigned_tree.changed_node, reassigned_tree.d, grammar)
        elseif reassigned_tree.transition == :bin2un
            # P of growing the new branch
            p_inv += tree_p(reassigned_tree.changed_node, reassigned_tree.d, grammar)
        else # reassigned_tree.transition == :same
            nothing 
        end 
    elseif mov == :re_feature
        tree = re_feature!(tree, grammar)
        p = p_inv = log(p_rf) + # P of movement = reassign feature
            log(1/nₜ) + # P of selecting this terminal node
            log(1/(length(terminal_is) - 1)) # P of choosing this new feature
    end 
    return TreeProposal(tree, mov, p, p_inv)
end 
