using ActiveSetPursuit
using Documenter

DocMeta.setdocmeta!(ActiveSetPursuit, :DocTestSetup, :(using ActiveSetPursuit); recursive=true)

makedocs(;
    modules=[ActiveSetPursuit],
    authors="tinatorabi <tntorabii@gmail.com> and contributors",
    sitename="ActiveSetPursuit.jl",
    format=Documenter.HTML(;
        canonical="https://MPF-Optimization-Laboratory.github.io/ActiveSetPursuit.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MPF-Optimization-Laboratory/ActiveSetPursuit.jl",
    devbranch="main",
)
