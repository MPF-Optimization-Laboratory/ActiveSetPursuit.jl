

function as_topy(A, b; min_lambda = 0.0, 
                       homotopy = true, 
                       kwargs...)
    m,n = size(A)
    bl = -ones(n)
    bu = +ones(n)
    active,state,xx,y,S,R, tracer = bpdual(A, b, min_lambda, bl, bu;
                             homotopy = homotopy, 
                             kwargs...)
    # BPdual's solution x is short. Make it full length.
    x = zeros(n)
    x[active] = xx
    return x, tracer
end
