struct ASPTracer{T}
    iteration::Vector{Int}
    lambda::Vector{T}
    solution::Vector{SparseVector{T}} 
end

Base.length(t::ASPTracer) = length(t.iteration)

function Base.getindex(t::ASPTracer, i::Integer)
    return t.solution[i], t.lambda[i]
end

Base.lastindex(t::ASPTracer) = lastindex(t.iteration)

@doc raw"""
```julia
    function bpdual(A, b, λin, bl, bu; kwargs...)``` 

    Solve the optimization problem:

    DP:
    ```math 
        \min_{y} \left( -b^T y + \frac{1}{2} λ y^T y \right)
    ```
        subject to:
     ```math 
        bl \leq A^T y \leq bu
        ```

    using given `A`, `b`, `bl`, `bu`, and `λ`. When `bl = -e` and `bu = e = ones(n, 1)`,
    DP is the dual Basis Pursuit problem:

    BPdual:

    ```math 
    \max_{y} \left( b^T y - \frac{1}{2} λ y^T y \right)
    ```
        subject to
        
    ```math
        \|A^T y\|_{\infty} \leq 1.
    ```

    * Input
    * `A` : `m`-by-`n` explicit matrix or operator.
    * `b` : `m`-vector.
    * `λin` : Nonnegative scalar.
    * `bl`, `bu` : `n`-vectors (bl lower bound, bu upper bound).
    * `active`, `state`, `y`, `S`, `R` : May be empty or output from `BPdual` with a previous value of `λ`.
    * `loglevel` : Logging level.
    * `coldstart` : Boolean indicating if a cold start should be used.
    * `homotopy` : Boolean indicating if homotopy should be used.
    * `λmin` : Minimum value for `λ`.
    * `trim` : Number of constraints to trim.
    * `itnMax` : Maximum number of iterations.
    * `feaTol` : Feasibility tolerance.
    * `optTol` : Optimality tolerance.
    * `gapTol` : Gap tolerance.
    * `pivTol` : Pivot tolerance.
    * `actMax` : Maximum number of active constraints.

    * Output
    * `tracer` : A structure to store trace information at each iteration of the optimization process.
        It contains:
            * `active::Vector{Int}`: The indices of the active constraints.
            * `activesoln::Vector{T}`: The solutions corresponding to the current active set.
            * `lambda::Vector{T}`: Lambda values.
            * `N::Int`: The total number of variables in the solution vector.
"""
function bpdual(
    A::Union{AbstractMatrix,AbstractLinearOperator},
    b::Vector,
    λin::Real,
    bl::Vector,
    bu::Vector;
    active::Union{Nothing, Vector{Int}} = nothing, ## maybe write as struct later
    state::Union{Nothing, Vector{Int}} = nothing,
    y::Union{Nothing, Vector{Float64}} = nothing,
    S = Matrix{Float64}(undef, size(A, 1), 0),
    R::Union{Nothing, Matrix{Float64}} = nothing,
    loglevel::Int = 1,
    coldstart::Bool = true,
    homotopy::Bool = false,
    λmin::Real = sqrt(eps(1.0)),
    trim::Int = 1,
    itnMax::Int = 10 * maximum(size(A)),
    feaTol::Real = 5e-05,
    optTol::Real = 1e-05,
    gapTol::Real = 1e-06,
    pivTol::Real = 1e-12,
    actMax::Real = Inf)

    # Start
    time0 = time()

    # ------------------------------------------------------------------
    # Grab input
    # ------------------------------------------------------------------
    m, n = size(A)

    tracer = ASPTracer(
        Int[],                  # iteration
        Float64[],              # lambda
        Vector{SparseVector{Float64}}() # now stores full sparse solutions
    )


    if coldstart || isnothing(active) || isnothing(state) || isnothing(y) || isnothing(S) || isnothing(R)
        active = Vector{Int}([])
        state = Vector{Int}(zeros(Int, n))
        y = Vector{Float64}(zeros(Float64, m))
        S = Matrix{Float64}(zeros(Float64, m, 0))
        R = Matrix{Float64}(zeros(Float64, 0, 0))
    end

    if homotopy
        z = A'*b
    else
        z = A'*y
    end

    tieTol = feaTol + 1e-8        # Perturbation to break ties
    λ = max(λin, λmin)
    if homotopy
        gapTol = 0
        lamFinal = λ
        λ = norm(z, Inf)
    end

    # ------------------------------------------------------------------
    # Print log header.
    # ------------------------------------------------------------------
    if loglevel > 0
        @info "-"^124
        @info @sprintf("%-30s : %-10d    %-30s : %-10.4e", "No. rows", m, "λ", λ)
        @info @sprintf("%-30s : %-10d    %-30s : %-10.1e", "No. columns", n, "Optimality tol", optTol)
        @info @sprintf("%-30s : %-10d    %-30s : %-10d", "Maximum iterations", itnMax, "Support trimming", trim)
        @info @sprintf("%-30s : %-10.1e    %-30s : %-10.1e", "Duality tol", gapTol, "Pivot tol", pivTol)
        @info "-"^124
    
        @info "-"^124
        @info @sprintf("| %-4s | %-8s | %-8s | %-6s | %-10s | %-10s | %-10s | %-7s | %-7s | %-9s |", 
                    "Itn", "Step", "Add/Drop", "Active", "||r||_2", "||x||_1", "Objective", "RelGap", "condS", "λ")
        @info "-"^124
    end

    # ------------------------------------------------------------------
    # Initialize local variables.
    # ------------------------------------------------------------------

    EXIT_INFO = Dict(
        :EXIT_OPTIMAL => "Optimal solution found -- full Newton step",
        :EXIT_TOO_MANY_ITNS => "Too many iterations",
        :EXIT_SINGULAR_LS => "Singular least-squares subproblem",
        :EXIT_INFEASIBLE => "Reached dual infeasible point",
        :EXIT_REQUESTED => "User requested exit",
        :EXIT_ACTMAX => "Max no. of active constraints reached",
        :EXIT_SMALL_DGAP => "Optimal solution found -- small duality gap",
        :EXIT_UNKNOWN => "unknown exit"
    )

    eFlag = :EXIT_UNKNOWN
    itn = 0
    step = 0.0
    p = 0      # index of added constraint
    q = 0      # index of deleted constraint
    svar = ""  # string value
    r = zeros(m)
    zerovec = zeros(n)
    numtrim = 0
    nprodA = 0
    nprodAt = 0

    # ------------------------------------------------------------------
    # Cold/warm-start initialization.
    # ------------------------------------------------------------------
    if coldstart
        x = zeros(0)
        if homotopy
            y = b / λ
            z = z / λ    
        else 
            y = zeros(m, 1)
            z = zeros(n, 1)
        end
    else
        g = b - λ*y  # Compute steepest-descent dir
        triminf(active, state, S, R, bl, bu, g)
        nact = length(active)
        x = zeros(nact)
        y = restorefeas(y, active, state, S, R, bl, bu)
        z = A'*y
        nprodAt += 1
    end

    sL, sU = infeasibilities(bl, bu, vec(z))
    inactive = abs.(state) .!= 1
    state[inactive .& (sL .> feaTol)] .= -2
    state[inactive .& (sU .> feaTol)] .= +2

    infeasible = any(abs.(state) .== 2)
    if infeasible
        eFlag = :EXIT_INFEASIBLE
    end

    # ------------------------------------------------------------------
    # Main loop.
    # ------------------------------------------------------------------
    while true
        sL, sU = infeasibilities(bl, bu, z)
        g = b - λ*y  # Steepest-descent direction

        if isempty(R)
            condS = 1
        else
            rmin = minimum(diag(R))
            rmax = maximum(diag(R))
            condS = rmax / rmin
        end

        if condS > 1e+10
            eFlag = :EXIT_SINGULAR_LS
            # Pad x with enough zeros to make it compatible with S.
            npad = size(S, 2) - size(x, 1)
            x = [x; zeros(npad)]
        else
            dx, dy = newtonstep(S, R, g, x, λ)
            x .+= dx
        end

        r = b - S*x

        # Print to log.
        yNorm = norm(y, 2)
        rNorm = norm(r, 2)
        xNorm = norm(x, 1)

        _, dObj, rGap = objectives(x, y, active, b, bl, bu, λ, rNorm, yNorm)
        nact = length(active)    

        if q != 0
            svar = @sprintf("%8i%s", q, "-")
        elseif !isempty(p)
            svar = @sprintf("%8i%s", p, " ")
        else
            svar = @sprintf("%8s%s", " ", " ")
        end

        if loglevel > 0
            @info @sprintf("| %4i | %8.1e | %8s | %6i | %10.4e | %10.4e | %10.4e | %7.1e | %7.1e | %9.3e |",
                itn, step, svar, nact, rNorm, xNorm, dObj, rGap, condS, λ)
        end

        # Check exit conditions.
        if eFlag == :EXIT_UNKNOWN && homotopy && λ <= lamFinal
            eFlag = :EXIT_OPTIMAL
        end

        if eFlag == :EXIT_UNKNOWN && rGap < gapTol && nact > 0
            eFlag = :EXIT_SMALL_DGAP
        end
        if eFlag == :EXIT_UNKNOWN && itn >= itnMax
            eFlag = :EXIT_TOO_MANY_ITNS
        end
        if eFlag == :EXIT_UNKNOWN && nact >= actMax
            eFlag = :EXIT_ACTMAX
        end

        # If this is an optimal solution, trim multipliers before exiting.
        if eFlag == :EXIT_OPTIMAL || eFlag == :EXIT_SMALL_DGAP
            if loglevel > 0
                @info "\nOptimal solution found. Trimming multipliers..."
            end
            g = b - λin*y
            trimx(x, S, R, active, state, g, b, λ, feaTol, optTol, loglevel)
            numtrim = nact - length(active)
            nact = length(active)
        end

        # Act on any live exit conditions.
        if eFlag != :EXIT_UNKNOWN
            break
        end

        # New iteration starts here.
        itn += 1
        p = q = 0

        if homotopy
            x, dy, dz, step, λ, p = htpynewlam(active, state, A, R, S, x, y, sL, sU, λ, lamFinal)
            nprodAt += 1
        else
            if norm(dy, Inf) < eps()        
                dz = zeros(n)
            else
                dz = A'*dy
                nprodAt += 1
            end
        end

        pL, stepL, pU, stepU = find_step(z, dz, bl, bu, state, tieTol, pivTol)
        if homotopy
            if isempty(p)
                hitBnd = false
            else
                hitBnd = true
                if p < 0
                    p = -p
                    state[p] = -1
                else
                    state[p] = +1
                end
            end
        else
            step = min(1.0, stepL, stepU)
            hitBnd = step < 1.0
            if hitBnd
                if step == stepL
                    p = pL
                    state[p] = -1
                else
                    p = pU
                    state[p] = 1
                end
            end
        end

        y += step * dy
        if mod(itn, 50) == 0
            z = A'*y
            nprodAt += 1
        else
            z += step * dz
        end

        if hitBnd
            zerovec[p] = 1
            a = A * zerovec
            nprodA += 1
            zerovec[p] = 0
            R = qraddcol(S, R, a)
            S = [S a]
            push!(active, p)
            push!(x, 0)
        else
            drop = false
            active = Int.(active)
            if length(active) > 0
                dropl = (state[active] .== -1) .& (x .> +optTol)
                dropu = (state[active] .== 1) .& (x .< -optTol)
                dropa = dropl .| dropu
                drop = any(dropa)
            end

            if drop
                nact = length(active)
                _, qa = findmax(abs.(x .* dropa))
                q = active[qa]
                state[q] = 0
                S = S[:, 1:nact .!= qa]
                deleteat!(active, qa)
                deleteat!(x, qa)
                R = qrdelcol(R, qa)
            else
                eFlag = :EXIT_OPTIMAL
            end
        end

        push!(tracer.iteration, itn)
        push!(tracer.lambda, λ)
        sparse_x_full = SparseVector(n, copy(active), copy(x))
        push!(tracer.solution, copy(sparse_x_full))
    end

    push!(tracer.iteration, itn)
    push!(tracer.lambda, λ)
    sparse_x_full = SparseVector(n, copy(active), copy(x))
    push!(tracer.solution, copy(sparse_x_full))

    tottime = time() - time0
    if loglevel > 0
        @info "\nEXIT BPdual -- $(EXIT_INFO[eFlag])\n\n"
        @info "No. significant nnz: $(sparsity(x)), Products with A: $nprodA"
        @info "No. trimmed nnz: $numtrim, Products with At: $nprodAt"
        @info "Solution time (sec): $tottime"
    end
    return tracer
end
