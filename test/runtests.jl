using ActiveSetPursuit
using Test

# List of test files and their respective test functions
tests = [
    ("test_bpdn.jl", :test_bpdn),
]

# Print header
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("   ActiveSetPursuit.jl: Start Tests   ")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")

@testset "ActiveSetPursuit.jl" begin
    for (test_file, test_function) in tests
        @testset "$(splitext(basename(test_file))[1])" begin
            include(test_file)   
            test_function = getfield(Main, test_function)   
            test_function()  
        end
    end
end
