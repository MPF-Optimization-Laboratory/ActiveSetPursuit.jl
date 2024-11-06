
# ------------------------------------------------------------------
#  Test BPDN & OMP
# ------------------------------------------------------------------ 

using Test, LinearAlgebra, Random, SparseArrays, ActiveSetPursuit, LinearOperators

function test_pursuits()
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
    tracer_bpdn = asp_bpdn(A, b, 0.0, loglevel =0);
    tracer_omp = asp_omp(A, b, 0.0, loglevel =0);
    xx_bpdn, _ = tracer_bpdn[end]
    xx_omp, _ = tracer_omp[end]

    pFeas_bpdn = norm(A * xx_bpdn - b, Inf) / max(1, norm(xx_bpdn, Inf))
    pFeas_omp = norm(A * xx_omp - b, Inf) / max(1, norm(xx_omp, Inf))

    @test pFeas_bpdn <= 1e-6
    @test pFeas_omp <= 1e-6
    # ------------------------
    # Linear operator
    # ------------------------

    OP = LinearOperator(A)
    b_op = OP * x
        
    # Solve the basis pursuit problem
    tracer_op_bpdn = asp_bpdn(OP, b_op, 0.0, loglevel =0);
    tracer_op_omp = asp_omp(OP, b_op, 0.0, loglevel =0);

    xx_op_bpdn, _ = tracer_op_bpdn[end]
    xx_op_omp, _ = tracer_op_omp[end]

    pFeas_bpdn_op = norm(OP * xx_op_bpdn - b_op, Inf) / max(1, norm(xx_op_bpdn, Inf))
    pFeas_omp_op = norm(OP * xx_op_omp - b_op, Inf) / max(1, norm(xx_op_omp, Inf))
    @test pFeas_bpdn_op <= 1e-6
    @test pFeas_omp_op <= 1e-6
end

Random.seed!(1234)

for ntest = 1:10
    test_pursuits()
end