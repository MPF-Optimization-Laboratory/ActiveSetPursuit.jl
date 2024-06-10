

function as_topy(A, b, lam)
    m,n = size(A)
    bl = -ones(n)
    bu = +ones(n)
    active,state,xx,y,S,R, tracer = bpdual(A, b, lam, bl, bu, homotopy = true)
    # BPdual's solution x is short. Make it full length.
    x = zeros(n)
    x[active] = xx
    return x, tracer
end
