# ------------------------------------------------------------------
# Helper functions 
# ------------------------------------------------------------------


## A) newtonstep
# ----------------------------------------------------------------------
# Compute a Newton step.  This is a step to a minimizer of the EQP
#
#   min   g'dy + 1/2 λ dy'dy   subj to   S'dy = 0.
#
# The optimality conditions are given by
#
#   [ -λ I   S ] [dy] = [ h ], with  h = b - lam y - S x, 
#   [   S'     ] [dx]   [ 0 ]
#
# where x is an estimate of the Lagrange multiplier. Thus, dx solves
# min ||S dx - h||. On input, g = b - lam y. Alternatively, solve the LS problem
# min ||S  x - g||.
# ----------------------------------------------------------------------

function newtonstep(S, R, g, x, λ)
    m, n = size(S)
    if m==0 || n==0
        dx = Float64[]
        dy = g/λ            # Steepest descent
        return (dx, dy)
    end
    h = g - S * x
    dx, dr = csne(R, S, vec(h))  # minimizes ||Sx-h||_2
    if m > n                # Overdetermined system
        dy = dr/λ           # dy is the scaled residual
    else                    # System is square or underdetermined
        dy = zeros(m)       # Anticipate that the residual is 0
    end
    return (dx, dy)
end # function newtonstep


## B) objectives


# ----------------------------------------------------------------------
# Compute the primal and dual objective values, and the duality gap:
#
#    DP:  minimize_y   - b'y  +  1/2 λ y'y
#         subject to   bl <= A'y <= bu
#
#    PP:  minimize_x   bl'neg(x) + bu'pos(x) + 1/2 λ y'y
#         subject to   Ax + λ y = b.
# ----------------------------------------------------------------------


function objectives(x, y, active, b, bl, bu, λ, rNorm, yNorm)

    bigNum = 1e20
    active = Int.(active)
    if isempty(x)
        blx = 0.
        bux = 0.
    else
        blx = bl[active]
        blx[blx .< -bigNum] .= 0
        blx = dot(blx, min.(x,0))

        bux = bu[active]
        bux[bux .>  bigNum] .= 0
        bux = dot(bux, max.(x,0))
    end

    if λ > eps(Float64)
        pObj = blx + bux + λ\(rNorm^2/2)   # primal objective
    else
        pObj = blx + bux
    end
    dObj  = λ*yNorm^2/2 - dot(b,y)         # dual   objective
    maxpd = max(max(1, pObj), dObj)
    rGap  = abs.(pObj+dObj)/maxpd               # relative duality gap

    return (pObj, dObj, rGap)

end # function objectives

## C)infeasibilities

# ----------------------------------------------------------------------
# Compute the infeasibility of z relative to bounds bl/bu.
#
#  (bl - z) <= 0  implies  z is   feasible wrt to lower bound
#           >  0  ...           infeasible ...
#  (z - bu) <= 0  implies  z is   feasible wrt to lower bound.
#           >  0  ...           infeasible ...
# ----------------------------------------------------------------------

function infeasibilities(bl, bu, z)
    sL = bl - z
    sU = z - bu
    return (sL, sU)
end


## D) restorefeas

# ----------------------------------------------------------------------
# Finds the least-norm change `dy` to `y` such that `y + dy` is feasible for the active set.

# minimize  ||dy||
# s.t.      S'(y + dy) = c

# where  c(i) = bl(i) if state(i) == -1
#             = bu(i) if state(i) == +1.
# This problem has optimality conditions
#      [ -I   S ] [dy] = [0]
#      [  S'    ] [ x]   [c - S'y],
# which we solve in two steps:
# 1. R'R x = c - S'y  (where R'R = S'S)
# 2. dy = Sx.
# ----------------------------------------------------------------------

function restorefeas(y, active, state, S, R, bl, bu)

    lbnd = (state[active] .== -1) .& (bl[active] .> -Inf)
    ubnd = (state[active] .== 1) .& (bu[active] .< Inf)
    c = bl[active] .* lbnd + bu[active] .* ubnd

    c .= c - S' * y
    x = R' \ c
    x = R \ x
    dy = S * x
    y .= y + dy
    return y
end



## E) trimx
# ----------------------------------------------------------------------
# Trim unneeded constraints from the active set.
# assumes that the current active set is optimal, ie,
# 1. x has the correct sign pattern, and
# 2. z := A'y is feasible.
# Keep trimming the active set until one of these conditions is
# violated. Condition 2 isn't checked directly. Instead, we simply check if
# the resulting change in y, dy, is small.  It would be more appropriate to
# check that dz := A'dy is small, but we want to avoid incurring additional
# products with A. (Implicitly we're assuming that A has a small norm.)
# ----------------------------------------------------------------------

function trimx(x,S,R,active,state,g,b,λ,featol,opttol,loglevel)

    k = 0
    nact = length(active)
    xabs = abs.(x)
    xmin, qa = findmin(xabs)
    gNorm = norm(g,Inf)

    while xmin < opttol

        e = sign.(x.*(xabs .> opttol)) # Signs of significant multipliers
        q = active[qa] # Index of the corresponding constraint
        a = S[:,qa]    # Save the col from S in case we need to add it back. 
        xsmall = x[qa] # Value of candidate multiplier

        # Trim quantities related to the small multiplier.
        deleteat!(e, qa)
        S = [S[:, 1:qa-1] S[:, qa+1:end]]
        deleteat!(active, qa)
        R = qrdelcol(R, qa)

        # Recompute the remaining multipliers and their signs.
        x, dy = csne(R, S, vec(g))           # min ||g - Sx||_2
        xabs = abs.(x)
        et = sign.(x.*(xabs .> opttol))
        dyNorm = norm(dy,Inf)/λ        # dy = (g - Sx) / lambda
    
        # Check if the trimmed active set is still optimal
        if any( et .!= e ) || (dyNorm / max(1,gNorm) > featol)
            R = qraddcol(S,R,a)
            S = [S a]
            push!(active, q)
            x = csne(R, S,vec(g))
            break
        end

        # The trimmed x is still optimal.
        k += 1
        nact -= 1
        state[q] = 0               # Mark this constraint as free.
        rNorm = norm(b - S*x)
        xNorm = norm(x,1)
        
        # Grab the next canddate multiplier.
        xmin, qa = findmin(xabs)
    end  
    return (x,S,R,active,state)
end


## F) triminf

# ----------------------------------------------------------------------
# Trim working constraints with "infinite" bounds.
# ----------------------------------------------------------------------

function triminf(active::Vector, state::Vector, S::Matrix, R::Matrix,
    bl::Vector, bu::Vector, g::Vector, b::Vector)

    bigbnd = 1e10
    nact = length(active)

    tlistbl = find( state[active] .== -1 & bl[active] .< -bigbnd )
    tlistbu = find( state[active] .== +1 & bu[active] .> +bigbnd )
    tlist   = [tlistbl; tlistbu]

    if isempty(tlist)
        return
    end

    for q in tlist
        qa = active[q]          
        nact = nact - 1 
        S = S[:,1:size(S,2) .!= qa] # Delete column from S
        deleteat!(active, qa)   # Delete index from active set
        R = qrdelcol(R, qa)     # Recompute new QR factorization
        state[q] = 0            # Mark constraint as free
        x = csne(R, S, g)       # Recompute multipliers

        rNorm = norm(b - S*x)
        xNorm = norm(x, 1)
        end

end


## G) find_step

# ----------------------------------------------------------------------
# Find step to first bound
# ----------------------------------------------------------------------

function find_step(z, dz, bl, bu, state, tieTol, pivTol)
    z, dz, bl, bu, state = vec(z), vec(dz), vec(bl), vec(bu), vec(state)
    
    # Conditions for movement
    moveToLower = (dz .< -pivTol) .& (state .== 0)
    moveToUpper = (dz .> pivTol) .& (state .== 0)

    stepL = stepU = Inf
    pL = pU = 0

    # Process lower bound candidates
    if any(moveToLower)
        validL = findall(moveToLower)
        sL = bl[validL] .- z[validL]
        tmpL = (sL .- tieTol) ./ dz[validL]
        stepL, indexL = findmin(tmpL)
        pL = validL[indexL]
        stepL = max(0, sL[indexL] / dz[validL][indexL])
    end

    # Process upper bound candidates
    if any(moveToUpper)
        validU = findall(moveToUpper)
        sU = z[validU] .- bu[validU]
        tmpU = -(sU .- tieTol) ./ dz[validU]
        stepU, indexU = findmin(tmpU)
        pU = validU[indexU]
        stepU = max(0, -sU[indexU] / dz[validU][indexU])
    end
    return (pL, stepL, pU, stepU)
end

## H) sparsity

function sparsity(x, threshold=0.9995)
    x = sort(abs.(x), rev=true)
    if sum(x) == 0
        return 0
    else
        x ./= sum(x)
        x = cumsum(x)
        return findfirst(≥(threshold), x)
    end
end


## I) htpynewlam
# ----------------------------------------------------------------------
# finds a new lambda for the homotopy method.
# ----------------------------------------------------------------------

function htpynewlam(active, state, A, R, S, x, y, s1, s2, λ, lamFinal)

    p = 0
    alfa = fill(Inf, 4)
    k = zeros(Int, 4)

    # Compute the search directions
    dx, dy = csne(R, S, y)
    dz = A' * dy

    active = Int.(active)

    # Initialize  
    i1 = 0
    i2 = 0

    # Find the largest allowable steps
    for i in eachindex(active)
        if state[active[i]] == 1 && dx[i] < -eps()
            step = -x[i] / dx[i]
            if step < alfa[1]
                alfa[1] = step
                i1 = i
                k[1] = active[i]
            end
        elseif state[active[i]] == -1 && dx[i] > eps()
            step = -x[i] / dx[i]
            if step < alfa[2]
                alfa[2] = step
                i2 = i
                k[2] = active[i]
            end
        end
    end

    # Steps along dy without violating constraints
    free = abs.(state) .!= 1
    r1 = dz .+ s1
    r2 = dz .- s2

    for i in eachindex(dz)
        if dz[i] < -eps() && free[i]
            step = λ * s1[i] / r1[i]
            if step < alfa[3]
                alfa[3] = step
                k[3] = i
            end
        elseif dz[i] > eps() && free[i]
            step = -λ * s2[i] / r2[i]
            if step < alfa[4]
                alfa[4] = step
                k[4] = i
            end
        end
    end

    # λ should never increase, ie, alphamin should always be nonnegative. 
    alfamin, kk = findmin(alfa)
    if alfamin < 0
        if alfamin > -1e-15
        alfamin = 0;
        else
            error(@sprintf("Change in lambda is negative: %s -- call for help!", alfamin))
        end
    end   

    alfa_max = λ - lamFinal
    dλ = min(alfamin, alfa_max)
    lamNew = λ - dλ

    force = lamNew > lamFinal 

    # Update variables
    x .+= dλ * dx
    if norm(dy) <= (1 + norm(y)) * sqrt(eps())
        step = 0
    else
        step = dλ / lamNew
    end

    if force
        if kk == 1       # x will go -
            x[i1] = -1
        elseif kk == 2   # x will go +
            x[i2] = 1
        elseif kk == 3   # A new lower-bound becomes active
            p = -k[kk]   # its index.
        else
            p = k[kk]
        end
    end    

    λ = lamNew
    if p ==0
        p = []
    end
    
    return x, dy, dz, step, λ, p
end