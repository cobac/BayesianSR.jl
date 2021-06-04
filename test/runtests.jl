using BayesianSR, Test

N_TEST = 20

using ExprRules, Random, LinearAlgebra, Distributions, Parameters, Symbolics
# using ExprTools
# using AbstractTrees

Random.seed!(2)
n = 100
k = 3
β = rand(Uniform(-2, 2), k + 1)
x = rand(Uniform(-10, 10), (n, k))
X = [ones(size(x, 1)) x]
ε = rand(Normal(0, 2), n)
y = X * β + ε

vargrammar = @grammar begin
    Real = x1 | x2 | x3
end 
fullgrammar = append!(deepcopy(BayesianSR.lineargrammar),
                      append!(deepcopy(BayesianSR.defaultgrammar), vargrammar))

hyper = Hyperparams()

function test_hyperparams(hyper::Hyperparams)
    @testset "Hyperparameters" begin
        names = hyper |> typeof |> fieldnames
        @test  length(names) == 4
        @test :k in names
        @test :σ²_prior in names
        @test :σ²_a_prior in names
        @test :σ²_b_prior in names
        @unpack k, σ²_prior, σ²_a_prior, σ²_b_prior = hyper
        @test typeof(k) <: Int
        @test typeof(σ²_prior) <: UnivariateDistribution
        @test typeof(σ²_a_prior) <: UnivariateDistribution
        @test typeof(σ²_b_prior) <: UnivariateDistribution
        σ² = rand(σ²_prior)
        @test typeof(σ²) <: AbstractFloat
        @test σ² >= 0
        σ²_a = rand(σ²_a_prior)
        @test typeof(σ²_a) <: AbstractFloat
        @test σ²_a >= 0
        σ²_b = rand(σ²_a_prior)
        @test typeof(σ²_b) <: AbstractFloat
        @test σ²_b >= 0
    end 
end 

test_hyperparams(hyper)

@testset "Grammars utils" begin
    node_types = BayesianSR.raw_nodetypes(fullgrammar)
    @test length(node_types) == length(fullgrammar)
    operator_is = findall(x -> x == 1 || x == 2, node_types)
    terminal_is = findall(x -> x == 0, node_types)
    deleteat!(terminal_is, 1)
    @test length(operator_is) + length(terminal_is) == length(node_types) - 1
    @test BayesianSR.operator_indices(fullgrammar) == operator_is
    @test BayesianSR.terminal_indices(fullgrammar) == terminal_is
end 

@testset "Grammars" begin
    var_operators = BayesianSR.raw_nodetypes(vargrammar)
    @test length(var_operators) == 3
    @test in(0, var_operators)
    @test maximum(var_operators) == 0

    xgrammar = BayesianSR.variablestogrammar(x)
    @test vargrammar.bytype == xgrammar.bytype
    @test vargrammar.childtypes == xgrammar.childtypes
    @test vargrammar.iseval == xgrammar.iseval
    @test vargrammar.isterminal == xgrammar.isterminal
    @test vargrammar.rules == xgrammar.rules
    @test vargrammar.types == xgrammar.types

    default_operators = BayesianSR.raw_nodetypes(BayesianSR.defaultgrammar)
    @test length(default_operators) == 6
    @test in(0, default_operators) == false
    @test in(1, default_operators)
    @test maximum(default_operators) == 2

    linear_operators = BayesianSR.nodetypes(BayesianSR.lineargrammar)
    @test length(linear_operators) == 2
    @test linear_operators[1] == -1
    @test linear_operators[2] == 1

    full_operators = BayesianSR.nodetypes(fullgrammar)
    @test length(full_operators) == length(var_operators) +
        length(default_operators) +
        length(linear_operators)
    @test maximum(full_operators) == 2
    @test minimum(full_operators) == -1
end

@testset "LinearCoef generation" begin
    lc = BayesianSR.LinearCoef(hyper)
    @test length(fieldnames(typeof(lc))) == 2
    @test typeof(lc.a) <: Float64
    @test typeof(lc.b) <: Float64
end 

@testset "Tree generation" begin
    Random.seed!(6)
    tree = RuleNode(fullgrammar, hyper)
    table = SymbolTable(fullgrammar, BayesianSR)
    i = 3
    table[:x1] = x[i, 1]
    table[:x2] = x[i, 2]
    table[:x3] = x[i, 3]
    eq = get_executable(tree, fullgrammar)
    answ = Core.eval(table, eq)
    @test length(answ) == 1
    @test isreal(answ)
    manual_answ = cos(x[3, 2]) / (-6.159091338126841 + 0.51775562953136 * x[3,1])
    @test answ ≈ manual_answ
    treex = BayesianSR.evaltree(tree, x, fullgrammar, table)
    @test length(treex) == size(x, 1)
end

@testset "Describe trees" begin
    tree = RuleNode(fullgrammar, hyper)
    nodes = BayesianSR.flatten(tree)
    @test length(nodes) == BayesianSR.n_operators(tree, fullgrammar) +
        BayesianSR.n_terminals(tree, fullgrammar)
    root = RuleNode(3, [RuleNode(2, [RuleNode(1), RuleNode(9)]),
                        RuleNode(10)])
    @test BayesianSR.n_operators(root, fullgrammar) == 2
    @test BayesianSR.n_linear_operators(root) == 1
    @test BayesianSR.n_terminals(root, fullgrammar) == 2
    @test BayesianSR.n_candidates(root, fullgrammar) == 2
end 

function test_tree(tree::RuleNode)
    @testset "Node sampling" begin
        terminal = BayesianSR.sampleterminal(tree, fullgrammar)
        terminal = get(tree, terminal).ind
        @test nchildren(fullgrammar, terminal) == 0
        if length(BayesianSR.flatten(tree)) > 1
            operator = BayesianSR.sampleoperator(tree, fullgrammar)
            operator = get(tree, operator).ind
            @test in(nchildren(fullgrammar, operator), [1,2])
        end 
    end 
    @testset "Linear operators and coefficients" begin
        tree = deepcopy(tree)
        queue = [tree]
        while !isempty(queue)
            current = queue[1]
            # No loose coefficients
            @test current.ind != 1
            # All linear operator has coefficients
            if current.ind == 2
                @test current.children[1].ind == 1                
                @test typeof(current.children[1]._val) <: BayesianSR.LinearCoef
                popfirst!(current.children)
            end 
            append!(queue, current.children)
            deleteat!(queue, 1)
        end
    end 
end 

@testset "Random node initialization" begin
    tree = RuleNode(fullgrammar, hyper)
    test_tree(tree)
end 

function test_sample(sample::BayesianSR.Sample)
    @testset "Samples" begin
        @test length(sample.trees) == k
        for tree in sample.trees
            test_tree(tree)
        end 
        @test length(sample.β) == k + 1
        key_names = keys(sample.σ²)
        @test length(key_names) == 3
        @test :σ² in key_names
        @test :σ²_a in key_names
        @test :σ²_b in key_names
    end 
end 

@testset "Random sample initialization" begin
    Random.seed!(1)
    sample = BayesianSR.Sample(fullgrammar, hyper)
    test_sample(sample)
    @test all(iszero.(BayesianSR.evalmodel(sample, x, fullgrammar)))
    @test maximum(sample.β) == 0
    BayesianSR.optimβ!(sample, x, y, fullgrammar)
    @test length(sample.β) == k + 1
    @test in(0, sample.β) == false
    @test all(isnan.(BayesianSR.evalmodel(sample, x, fullgrammar))) == false
end 

function test_chain(chain::Chain; initial=true)
    initial && @test length(chain) == 1
    @test length(chain) == length(chain.samples)
    initial && @test chain.samples[1] == chain.samples[end]
    initial ? test_sample(chain.samples[1]) : test_sample(chain.samples[end])

    stat_keys = keys(chain.stats)
    @test length(stat_keys) == 3
    @test :lastj in stat_keys
    @test chain.stats[:lastj] <= chain.hyper.k
    @test :accepted in stat_keys
    @test chain.stats[:accepted] <= length(chain)
    @test :last_sample in stat_keys
    @test chain.stats[:last_sample] == length(chain)
    test_hyperparams(chain.hyper)
end 

@testset "Random Chain initialization" begin
    k = 3
    chain = Chain(x, y)
    @test chain[1] == chain.samples[1]
    test_chain(chain)
    chain = Chain(x, y, hyper=hyper)
    test_chain(chain)
    chain = Chain(x, y, operators=deepcopy(BayesianSR.defaultgrammar))
    test_chain(chain)
    chain = Chain(x, y, operators=deepcopy(BayesianSR.defaultgrammar), hyper=hyper)
    test_chain(chain)
end 

@testset "Utils" begin 
    @testset "flatten()" begin
        node = RuleNode(2, [RuleNode(3, [RuleNode(4), RuleNode(5)])])
        flat = BayesianSR.flatten(node)
        @test length(flat) == 4
        @test in(1, flat) == false
        @test all(in.(2:5, flat))
    end 
    @testset "sampleterminal()" begin
        for i in 1:N_TEST
            node = RuleNode(fullgrammar, hyper)
            loc = BayesianSR.sampleterminal(node, fullgrammar)
            node = get(node, loc)
            @test node.ind in BayesianSR.terminal_indices(fullgrammar)
        end 
    end 
    @testset "sampleoperator()" begin
        for i in 1:N_TEST
            node = RuleNode(fullgrammar, hyper)
            loc = BayesianSR.sampleoperator(node, fullgrammar)
            node = get(node, loc)
            @test node.ind in BayesianSR.operator_indices(fullgrammar)
        end 
    end 
    @testset "iscandidate()" begin
        node1 = RuleNode(3, [RuleNode(9), RuleNode(9)])
        @test BayesianSR.iscandidate(node1, node1, fullgrammar) == false
        node2 = RuleNode(3, [RuleNode(3, [RuleNode(9), RuleNode(9)]), RuleNode(9)])
        @test BayesianSR.iscandidate(node2, node2, fullgrammar)
        node3 = RuleNode(3, [RuleNode(3, [RuleNode(9), RuleNode(9)]),
                             RuleNode(3, [RuleNode(9), RuleNode(9)])])
        @test BayesianSR.iscandidate(node3, node3, fullgrammar)
        @test BayesianSR.iscandidate(RuleNode(9), node3, fullgrammar) == false
        @test BayesianSR.iscandidate(RuleNode(3, [RuleNode(9), RuleNode(9)]), node3, fullgrammar)
    end

    @testset "samplecandidate()" begin
        node1 = RuleNode(3, [RuleNode(9), RuleNode(9)])
        @test_throws ErrorException BayesianSR.samplecandidate(node1, fullgrammar)
        node2 = RuleNode(3, [RuleNode(3, [RuleNode(9), RuleNode(9)]), RuleNode(9)])
        for i in 1:10
            @test BayesianSR.samplecandidate(node2, fullgrammar).i in [0, 1]
        end 
        node3 = RuleNode(3, [RuleNode(3, [RuleNode(9), RuleNode(9)]),
                             RuleNode(3, [RuleNode(9), RuleNode(9)])])
        for i in 1:10
            @test BayesianSR.samplecandidate(node3, fullgrammar).i in [0, 1, 2]
        end 
    end 
end 

@testset "Tree movements" begin
    # TODO: Test edge cases with linear coef
    @testset "grow!()" begin
        for _ in 1:N_TEST
            node = RuleNode(fullgrammar, hyper)
            old_length = length(BayesianSR.flatten(node))
            proposal = BayesianSR.grow!(node, fullgrammar, hyper)
            new_length = length(BayesianSR.flatten(proposal.tree))
            @test new_length >= old_length
            test_tree(proposal.tree)
        end 
    end 

    @testset "prune!()" begin
        for _ in 1:N_TEST
            node = RuleNode(fullgrammar, hyper)
            old_length = length(BayesianSR.flatten(node))
            proposal = BayesianSR.prune!(node, fullgrammar)
            new_length = length(BayesianSR.flatten(proposal.tree))
            @test new_length < old_length
            test_tree(proposal.tree)
            @test proposal.changed_node.ind in BayesianSR.operator_indices(fullgrammar)
        end 
    end 

    @testset "delete!()" begin
        function new_deleteable_node()
            node = BayesianSR.RuleNode(fullgrammar, hyper)
            try BayesianSR.samplecandidate(node, fullgrammar)
            catch e
                node = new_deleteable_node()
            end 
            return node
        end 
        node = new_deleteable_node()
        old_length = length(BayesianSR.flatten(node))
        proposal = BayesianSR.delete!(node, fullgrammar)
        new_length = length(BayesianSR.flatten(proposal.tree))
        @test new_length < old_length
        test_tree(proposal.tree)
        for i in 1:10
            node2 = RuleNode(3, [RuleNode(4, [RuleNode(9), RuleNode(9)]), RuleNode(11)])
            proposal2 = BayesianSR.delete!(node2, fullgrammar)
            if proposal2.dropped_node == RuleNode(9)
                @test proposal2.p_child == 0.5
            else 
                @test proposal2.p_child == 1
            end 
        end 
        node3 = RuleNode(3, [RuleNode(4, [RuleNode(9), RuleNode(9)]),
                             RuleNode(5, [RuleNode(9), RuleNode(9)])])
        proposal3 = BayesianSR.delete!(node3, fullgrammar)
        @test proposal3.p_child == 0.5
    end 

    @testset "insert_node!()" begin
        for _ in 1:N_TEST
            node = BayesianSR.RuleNode(fullgrammar, hyper)
            old_length = length(BayesianSR.flatten(node))
            proposal = BayesianSR.insert_node!(node, fullgrammar, hyper)
            new_length = length(BayesianSR.flatten(proposal.tree))
            @test new_length > old_length
            test_tree(proposal.tree)
        end 
    end 

    @testset "re_operator!()" begin
        for _ in 1:N_TEST
            node = BayesianSR.RuleNode(fullgrammar, hyper)
            old_node = deepcopy(node)
            old_length = length(BayesianSR.flatten(node))
            proposal = BayesianSR.re_operator!(node, fullgrammar, hyper)
            node = proposal.tree
            new_length = length(BayesianSR.flatten(node))
            @test node != old_node
            test_tree(node)
        end 
    end 

    @testset "re_feature!()" begin
        for _ in 1:N_TEST
            node = BayesianSR.RuleNode(fullgrammar, hyper)
            old_node = deepcopy(node)
            old_length = length(BayesianSR.flatten(node))
            node = BayesianSR.re_feature!(node, fullgrammar)
            new_length = length(BayesianSR.flatten(node))
            @test new_length == old_length
            @test node != old_node
            test_tree(node)
        end 
    end 
end 

@testset "Tree prior" begin
    node = RuleNode(7, [RuleNode(8, [RuleNode(2, [RuleNode(1), RuleNode(12)])])])
    manual_flatten = [BayesianSR.IndAndCount(7, 1),
                      BayesianSR.IndAndCount(8, 2),
                      BayesianSR.IndAndCount(2, 3),
                      BayesianSR.IndAndCount(12, 4)]
    @test BayesianSR.flatten_with_depth(node, 1) == manual_flatten
    manual_flatten2 = [BayesianSR.IndAndCount(7, 2),
                       BayesianSR.IndAndCount(8, 3),
                       BayesianSR.IndAndCount(2, 4),
                       BayesianSR.IndAndCount(12, 5)]
    @test BayesianSR.flatten_with_depth(node, 2) == manual_flatten2
    manual_p = log(2 / (1 + 1)) +
        log(2 / (1 + 2)) +
        log(2 / (1 + 3)) + 
        log(1 / 7) * 3 + 
        log(1 - 2 / (1 + 4)) + 
        log(1 / 3)
    @test BayesianSR.tree_p(node, fullgrammar) ≈ manual_p
    manual_p2 = log(2 / (1 + 2)) +
        log(2 / (1 + 3)) +
        log(2 / (1 + 4)) + 
        log(1 / 7) * 3 + 
        log(1 - 2 / (1 + 5)) + 
        log(1 / 3)
    @test BayesianSR.tree_p(node, 2, fullgrammar) ≈ manual_p2
end 

@testset "LinearCoef Proposals" begin
    @testset "Recover" begin
        Random.seed!(1)
        node = RuleNode(fullgrammar, hyper)
        θ = BayesianSR.recover_LinearCoef(node)
        @test length(θ.a) == length(θ.b) == 0

        Random.seed!(2)
        node = RuleNode(fullgrammar, hyper)
        θ = BayesianSR.recover_LinearCoef(node)
        @test length(θ.a) == length(θ.b) == 1
        @test θ.a[1] ≈ 15.632280168597802
        @test θ.b[1] ≈ 2.687852482764168

        Random.seed!(104)
        node = RuleNode(fullgrammar, hyper)
        θ = BayesianSR.recover_LinearCoef(node)
        @test length(θ.a) == length(θ.b) == 2
        @test θ.a == [-0.908098317686729, 6.7865155459951945]
        @test θ.b == [-0.6969881650341446, -6.4827857902895625]
    end 

    @testset "No RJMCMC" begin
        Random.seed!(104)
        node = RuleNode(fullgrammar, hyper)
        θ_old = BayesianSR.recover_LinearCoef(node)
        θ = BayesianSR.propose_LinearCoef!(node, 1., 1.)
        @test θ_old.a != θ.b
        @test θ_old.b != θ.b
        @test node.children[1].children[1]._val == BayesianSR.LinearCoef(θ.a[1], θ.b[1])
    end 

    @testset "Expansion" begin
        θ_old = BayesianSR.SetLinearCoef(rand(3), rand(3))
        old_node = RuleNode(1, BayesianSR.LinearCoef(θ_old.a[1], θ_old.b[1]),
                            [RuleNode(1, BayesianSR.LinearCoef(θ_old.a[2], θ_old.b[2])),
                             RuleNode(1, BayesianSR.LinearCoef(θ_old.a[3], θ_old.b[3]))])
        node = RuleNode(2, [deepcopy(old_node),
                            RuleNode(1, BayesianSR.LinearCoef(hyper)),
                            RuleNode(1, BayesianSR.LinearCoef(hyper)),
                            RuleNode(1, BayesianSR.LinearCoef(hyper)),
                            RuleNode(1, BayesianSR.LinearCoef(hyper)),
                            RuleNode(1, BayesianSR.LinearCoef(hyper)),
                            RuleNode(1, BayesianSR.LinearCoef(hyper)),
                            ])
        θ, U, U⁺ = BayesianSR.propose_LinearCoef!(node, 1., 1., θ_old, :expansion)
        @test length(θ.a) == length(θ.b) == 9
        @test length(U.a) == length(U.b) == 9
        @test length(U⁺.a) == length(U⁺.b) == length(θ_old.a)
        @test length(θ.a) + length(U⁺.a) == length(θ_old.a) + length(U.a)
    end 

    @testset "Shrinkage" begin
        old_node = RuleNode(2, [RuleNode(1, BayesianSR.LinearCoef(hyper)),
                                RuleNode(1, BayesianSR.LinearCoef(hyper)),
                                RuleNode(1, BayesianSR.LinearCoef(hyper)),
                                RuleNode(1, BayesianSR.LinearCoef(hyper)),
                                ])
        θ_old = BayesianSR.recover_LinearCoef(old_node)
        node =  RuleNode(1, BayesianSR.LinearCoef(θ_old.a[1], θ_old.b[1]),
                         [RuleNode(1, BayesianSR.LinearCoef(θ_old.a[2], θ_old.b[2])),
                          RuleNode(1, BayesianSR.LinearCoef(θ_old.a[1], θ_old.b[1]))])
        θ, U, U⁺ = BayesianSR.propose_LinearCoef!(node, 1., 1., θ_old, :shrinkage)
        @test length(θ.a) == length(θ.b) == 3
        @test length(U.a) == length(U.b) == 3
        @test length(U⁺.a) == length(U⁺.b) == length(θ_old.a)
        @test length(θ.a) + length(U⁺.a) == length(θ_old.a) + length(U.a)
    end 
end 

@testset "Interface" begin
    chain = Chain(x, y)
    @variables x1, x2, x3
    ex = get_function(chain)
    f = build_function(ex, [x1, x2, x3], expression = Val{false})
    ŷ = evalmodel(chain)
    for i in 1:n
        @test f(@view chain.x[i, :]) ≈ ŷ[i]
    end 
end 

@testset "MCMC: one chain" begin
    for _ in 1:N_TEST
        chain = Chain(x, y)
        test_chain(chain)
        n = 100
        mcmc!(chain, n, verbose=false)
        test_chain(chain, initial=false)
        @test length(chain) == n + 1
        @test all([isassigned(chain.samples, i) for i in 1:length(chain)])
    end 
end 
