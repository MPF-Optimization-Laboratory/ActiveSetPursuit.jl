
# ------------------------------------------------------------------
#  Test Basis pursuit
# ------------------------------------------------------------------ 

using Test, LinearAlgebra, Random, SparseArrays, ActiveSetPursuit, LinearOperators

function test_bpdn()
    m = 600
    n = 2560
    k = 100
    
    # Generate sparse solution
    p = randperm(n)[1:k]  # Position of nonzeros in x
    x = zeros(n)
    x[p] .= randn(k)

    A = randn(m, n)

    # Compute the RHS vector
    b = A * x
    
    # Solve the basis pursuit problem
    tracer = asp_bpdn(A, b, 0.0, loglevel =0, refactor_freq =20);
    

    xx, _ = tracer[end]
    x_recovered = sparsevec(xx.active, xx.values, n)

    pFeas = norm(A * x_recovered - b, Inf) / max(1, norm(x_recovered, Inf))
    @test pFeas <= 1e-6

    # ------------------------
    # Linear operator
    # ------------------------

    OP = LinearOperator(A)
    b_op = OP * x
        
    # Solve the basis pursuit problem
    tracer_op = asp_bpdn(OP, b_op, 0.0, loglevel =0);

    xx_op, _ = tracer_op[end]
    xop_recovered = sparsevec(xx_op.active, xx_op.values, n)

    pFeas_op = norm(OP * xop_recovered - b_op, Inf) / max(1, norm(xop_recovered, Inf))
    @test pFeas_op <= 1e-6
end

Random.seed!(1234)

for ntest = 1:10
    test_bpdn()
end