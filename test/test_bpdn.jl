
# ------------------------------------------------------------------
#  Test Basis pursuit
# ------------------------------------------------------------------ 

using Test, LinearAlgebra, Random, SparseArrays, ActiveSetPursuit, LinearOperators

function test_bpdn()
    m = 600
    n = 2560
    k = 20
    
    # Generate sparse solution
    p = randperm(n)[1:k]  # Position of nonzeros in x
    x = zeros(n)
    x[p] .= randn(k)

    A = randn(m, n)

    # Compute the RHS vector
    b = A * x
    
    # Solve the basis pursuit problem
    tracer = asp_bpdn(A, b, 0.0, loglevel =0);

    xx, λ = tracer[end]
    pFeas = norm(A * xx - b, Inf) / max(1, norm(xx, Inf))
    @test pFeas <= 1e-6

    # ------------------------
    # Linear operator
    # ------------------------

    OP = LinearOperator(A)
    b_op = OP * x
        
    # Solve the basis pursuit problem
    tracer_op = asp_bpdn(OP, b_op, 0.0, loglevel =0);

    xx_op, λ_op = tracer_op[end]
    pFeas_op = norm(OP * xx_op - b_op, Inf) / max(1, norm(xx_op, Inf))
    @test pFeas_op <= 1e-6
end

Random.seed!(1234)

for ntest = 1:10
    test_bpdn()
end