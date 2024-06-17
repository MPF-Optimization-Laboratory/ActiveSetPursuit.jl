struct OMPTracer{T}
    iteration::Vector{Int}
    active::Vector{Vector{Int}}
    activesoln::Vector{Vector{T}}
    N::Int
end

Base.length(t::OMPTracer) = length(t.iteration)

function Base.getindex(t::OMPTracer, i::Integer)
    as = t.active[i]
    x = zeros(t.N)
    x[as] = t.activesoln[i]
    return x 
end

Base.lastindex(t::OMPTracer) = lastindex(t.active)

function asp_omp(
    A::Union{AbstractMatrix, AbstractLinearOperator},
    b::Vector,
    λin::Real,
    active::Union{Nothing, Vector{Int}} = nothing,  
    state::Union{Nothing, Vector{Int}} = nothing,
    S::Matrix{Float64} = Matrix{Float64}(undef, size(A, 1), 0),
    R::Union{Nothing, Matrix{Float64}} = nothing,
    loglevel::Int = 1,
    λmin::Real = sqrt(eps(1.0)),
    itnMax::Int = 10 * maximum(size(A)),
    feaTol::Real = 5e-05,
    optTol::Real = 1e-05,
    gapTol::Real = 1e-06,
    pivTol::Real = 1e-12,
    actMax::Real = Inf)
    
    """
        Orthogonal matching pursuit for sparse Ax=b

        asp_omp applies the orthogonal matching pursuit (OMP) algorithm to
        estimate a sparse solution of the underdetermined system Ax=b.

        (BP)   minimize_x  ||x||_1  subject to  Ax = b.

        asp_omp(A, b, λ) solves the basis pursuit problem.
    """
    # Start the clock and size up the problem.
    time0 = time()

    z = A' * b

    m = length(b)
    n = length(z)
    nprodA = 0
    nprodAt = 1

    # Initialize the tracer
    tracer = OMPTracer(
        Int[],                  # iteration
        Vector{Vector{Int}}(),  # active
        Vector{Vector{Float64}}(), # activesoln
        n                       # N
    )

    # Print log header.

    if loglevel > 0
        @info "-"^124
        @info @sprintf("%-30s : %-10d    %-30s : %-10.4e", "No. rows", m, "λ", λin)
        @info @sprintf("%-30s : %-10d    %-30s : %-10.1e", "No. columns", n, "Optimality tol", optTol)
        @info @sprintf("%-30s : %-10d    %-30s : %-10.1e", "Maximum iterations", itnMax, "Duality tol", gapTol)
        @info "-"^124
    end

    # Initialize local variables.
    EXIT_INFO = Dict(
        :EXIT_OPTIMAL => "Optimal solution found -- full Newton step",
        :EXIT_TOO_MANY_ITNS =>  "Too many iterations",
        :EXIT_SINGULAR_LS => "Singular least-squares subproblem",
        :EXIT_LAMBDA => "Reached minimum value of lambda",
        :EXIT_RHS_ZERO => "b = 0. The solution is x = 0",
        :EXIT_UNCONSTRAINED => "Unconstrained solution r = b is optimal",
        :EXIT_UNKNOWN => "unknown exit"
    )

    itn = 0
    eFlag = :EXIT_UNKNOWN
    x = zeros(Float64, 0)
    zerovec = zeros(Float64, n)
    p = 0

    # Quick exit if the RHS is zero.
    if norm(b, Inf) == 0
        r = zeros(m)
        eFlag = :EXIT_RHS_ZERO
    end

    # Solution is unconstrained for lambda large.
    zmax = norm(z, Inf)
    if eFlag == :EXIT_UNKNOWN && zmax < λin
        r = b
        eFlag = :EXIT_UNCONSTRAINED
    end

    if eFlag != :EXIT_UNKNOWN || active === nothing
        active = Vector{Int}([])
    end
    if state === nothing
        state = zeros(Int, n)
    end
    if R === nothing
        R = Matrix{Float64}(undef, 0, 0)
    end

    @info @sprintf("%4s  %8s %12s %12s %12s", "Itn", "Var", "λ", "rNorm", "xNorm")


    # Main loop.
    while true
        # Compute dual obj gradient g, search direction dy, and residual r.
        if itn == 0
            x = Float64[]
            r = b
            z = A' * r
            nprodAt += 1
            zmax = norm(z, Inf)
        else
            x,y = csne(R, S, vec(b))
            if norm(x, Inf) > 1e12
                eFlag = :EXIT_SINGULAR_LS
                break
            end

            Sx = S * x
            r = b - Sx
        end

        rNorm = norm(r, 2)
        xNorm = norm(x, 1)

@info @sprintf("%4i  %8i %12.5e %12.5e %12.5e", itn, p, zmax, rNorm, xNorm)

        # Check exit conditions.
        if eFlag != :EXIT_UNKNOWN
            # Already set. Don't test the other exits.
        elseif zmax <= λin
            eFlag = :EXIT_LAMBDA
        elseif rNorm <= optTol
            eFlag = :EXIT_OPTIMAL
        elseif itn >= itnMax
            eFlag = :EXIT_TOO_MANY_ITNS
        end
        
        if eFlag != :EXIT_UNKNOWN
            break
        end

        # New iteration starts here.
        itn += 1

        # Find step to the nearest inactive constraint
        z = A' * r

        nprodAt += 1
        zmax, p = findmax(abs.(z))

        if z[p] < 0
            state[p] = -1
        else
            state[p] = 1
        end

        zerovec[p] = 1   # Extract a = A[:, p]
        a = A * zerovec

        nprodA += 1
        zerovec[p] = 0

        R = qraddcol(S, R, a)  # Update R
        S = hcat(S, a)  # Expand S, active
        push!(active, p)

        push!(tracer.iteration, itn)
        push!(tracer.active, copy(active))
        push!(tracer.activesoln, copy(x))
    end #while true

    push!(tracer.iteration, itn)
    push!(tracer.active, copy(active))
    push!(tracer.activesoln, copy(x))

    tottime = time() - time0
    if loglevel > 0
        @info @sprintf("\nEXIT BPdual -- %s\n", EXIT_INFO[eFlag])
        @info @sprintf("%-20s: %8i", "Products with A", nprodA)
        @info @sprintf("%-20s: %8i", "Products with At", nprodAt)
        @info @sprintf("%-20s: %8.1e", "Solution time (sec)", tottime)
        @info "\n"
    end
    return tracer
end