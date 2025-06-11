using ActiveSetPursuit
using Test


# Print header
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("    ActiveSetPursuit.jl: Start Tests     ")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")

@testset "ActiveSetPursuit.jl" begin
    @testset "BPDN" begin include("test_bpdn.jl") end
    @testset "Recover decaying coefficients" begin include("test_recover_decaying.jl") end
    @testset "OMP" begin include("test_omp.jl") end
    @testset "triminf" begin include("test_triminf.jl") end
end

