push!(LOAD_PATH,"../src/")

using Documenter, DocumenterInterLinks, UTrack, CUDA

links = InterLinks(
    "CUDA" => "https://cuda.juliagpu.org/stable/objects.inv"
)

pages = [
    "Overview" => "index.md",
    "User Manual" => "manual/index.md",
    "Developer Documentation" => "documentation.md",
]

makedocs(sitename="UTrack.jl", plugins=[links], pages=pages)