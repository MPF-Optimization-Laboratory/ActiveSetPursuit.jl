using Test
using LinearAlgebra
using ActiveSetPursuit: triminf!, trimx
using Random

# Test A1: No constraints to remove
active1 = [1, 2]
state1  = [0, 0]
m, n = 3, 2
S1 = rand(m, n)
R1 = Matrix{Float64}(I, n, n)   
bl1 = [0.0, 0.0]
bu1 = [0.0, 0.0]
g = zeros(m)
b = zeros(m)

active_out1, state_out1, S_out1, R_out1 =
    triminf!(copy(active1), copy(state1), copy(S1), copy(R1), bl1, bu1)

@test active_out1 == active1
@test state_out1 == state1
@test S_out1 == S1

# Test A2: All constraints out of bounds
active2 = [1, 2]
state2  = [-1, 1]
S2 = rand(m, n)
R2 = Matrix{Float64}(I, n, n)
bl2 = [-1e11, 0.0]
bu2 = [0.0, 1e11]

active_out2, state_out2, S_out2, R_out2 =
    triminf!(copy(active2), copy(state2), copy(S2), copy(R2), bl2, bu2)

@test isempty(active_out2)
@test state_out2 == [0, 0]
@test size(S_out2) == (m, 0)
@test size(R_out2) == (0, 0)
@test size(R_out2,1) == size(R_out2,2)  
@test istriu(R_out2) 



###### Test B1: Test trimx

@testset "trimx" begin

    # last two multipliers are negligible
    Random.seed!(42)
    m = 10

    S_base = randn(m, 4)
    a_weak1 = randn(m) * 1e-8
    a_weak2 = randn(m) * 1e-8
    S = [S_base a_weak1 a_weak2]

    g = S_base * [2.0, -1.5, 0.8, -3.1]
    x_init = S \ g
    R = Matrix(qr(S).R)

    active   = [1, 2, 3, 4, 5, 6]
    state    = ones(Int, 6) # all active 
    λ        = 1.0
    featol   = 1e-4
    opttol   = 1e-6

    x_out, active_out = trimx(x_init, S, R, active, state, g, λ, featol, opttol)

    @test length(active_out) < 6
    @test all(abs.(x_out) .>= opttol)
    @test 5 ∉ active_out
    @test 6 ∉ active_out
    @test state[5] == 0
    @test state[6] == 0
end