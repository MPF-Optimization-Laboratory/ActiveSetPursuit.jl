# ActiveSetPursuit

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MPF-Optimization-Laboratory.github.io/ActiveSetPursuit.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MPF-Optimization-Laboratory.github.io/ActiveSetPursuit.jl/dev/)
[![Build Status](https://github.com/MPF-Optimization-Laboratory/ActiveSetPursuit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MPF-Optimization-Laboratory/ActiveSetPursuit.jl/actions/workflows/CI.yml?query=branch%3Amain)


## ActiveSetPursuit

`ActiveSetPursuit` is a Julia package providing a framework for solving several variations of the sparse optimization problem:

$$\underset{x}{\text{minimize}} \hspace{0.5em}  \lambda \||x\||_1 + \frac{1}{2} \||Ax - b\||^2_2$$

### Implemented Algorithms:
- Basis Pursuit Denoising
- Orthogonal Matching Pursuit
- Homotopy Basis Pursuit Denoising

### Installation:

The package can be installed as follows:
```jlcon
julia> ] add ComplexElliptic

```



## Example Usage

The `asp_bpdn` function within the `ActiveSetPursuit` package efficiently solves the basis pursuit denoising problem.

Ensure that the following inputs are defined before running the `asp_bpdn` function:
- **`A`**: The matrix or operator, size `m`-by-`n`.
- **`b`**: The vector of observations or measurements, size `m`.
- **`λin`**: A nonnegative scalar that serves as the regularization parameter.

To solve the basis pursuit denoising problem, execute the following command in Julia:

```julia
julia> tracer = asp_bpdn(A, b, λin)
```
After completing the optimization process, the solution and the regularization parameter at each iteration `itn` can be accessed as follows:
```jlcon
julia> xx, λ = tracer[itn]
```
