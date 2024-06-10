
# ------------------------------------------------------------------
#  Test Basis pursuit
# ------------------------------------------------------------------ 

using Test, LinearAlgebra, Random, SparseArrays, ActiveSetPursuit

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
    bl = -ones(n)
    bu = +ones(n)
    
    # Solve the basis pursuit problem
    active,state,xs,y,S,R,tracer = bpdual(A, b, 0., bl, bu, homotopy = false, loglevel =0)
    xx = zeros(n)
    xx[Int.(active)] = xs
    
    pFeas = norm(A * xx - b, Inf) / max(1, norm(xx, Inf))
    dFeas = max(0, norm(A' * y, Inf) - 1)
    dComp = abs(norm(xx, 1) - dot(b, y))
    
    @test pFeas <= 1e-6
    @test dFeas <= 1e-6
    @test dComp <= 1e-6
end


for ntest = 1:10
    test_bpdn()
end