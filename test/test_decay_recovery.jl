# ------------------------------------------------------------------
#  Test Basis pursuit with decaying and permuted coefficients
# ------------------------------------------------------------------ 

using Test, LinearAlgebra, Random, SparseArrays, ActiveSetPursuit

function test_recover_decaying()
    m = 600
    n = 2560
    threshold_percentage = 0.9  # Threshold percentage of the norm

    # Generate solution with decaying coefficients and permutate
    x = randn(n) ./ (1:n).^2
    perm = randperm(n)
    x = x[perm]

    A = randn(m, n)

    # Compute the RHS vector
    b = A * x
    bl = -ones(n)
    bu = +ones(n)

    # Solve the basis pursuit problem
    xs, tracer = asp_homotopy(A, b, min_lambda = 0.0, itnMax =300,loglevel =0)

    sorted_indices = sortperm(abs.(x), rev=true)
    cumulative_norm = cumsum(abs.(x[sorted_indices]))
    total_norm = sum(abs.(x))

    threshold_norm = threshold_percentage * total_norm
    indices_to_recover = sorted_indices[findall(cumulative_norm .<= threshold_norm)]

    # Compare these indices with the recovered solution
    # Get the corresponding values in the recovered solution
    recovered_indices = sortperm(abs.(xs), rev=true)[1:length(indices_to_recover)]

    # Check if all the significant indices are correctly recovered
    indices_correct = all(in(indices_to_recover).(recovered_indices))

    values_correct = all((abs.(x[indices_to_recover]) .- abs.(xs[recovered_indices])) .< 1e-3)

    @test indices_correct
    @test values_correct
end

# # Run the tests
# for ntest = 1:10
#     test_recover_decaying()
# end
