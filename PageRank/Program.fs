(**
    Bryan Rowe
    UC Riverside
    "ALeaGPU PageRank"

    Run PageRank on the CPU and to launch PageRank on
    either the CPU or GPU (calling PageRank.fs function).

    Version 1.0
**)

open System
open System.IO

open System
open System.IO
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.IL
open NUnit.Framework

// Based on this graph: http://www.cis.temple.edu/~giorgio/cis307/readings/MapReduce/transitionGraph.png   
// Incoming edges.  There is an edge from the value to the index of the 2D array.
let incomingEdge = Array2D.create 6 2 -1

// Node id 0 has an edges 0 -> 3 and 0 -> 2
incomingEdge.[0,0] <- 3
incomingEdge.[0,1] <- 2
incomingEdge.[1,0] <- 0
incomingEdge.[2,0] <- 1
incomingEdge.[2,1] <- 5
incomingEdge.[3,0] <- 1
incomingEdge.[4,0] <- 1
incomingEdge.[4,1] <- 2
incomingEdge.[5,0] <- 1
incomingEdge.[5,1] <- 3

// 1D Array version for GPU execution
let incomingEdge1D = [|3;2;0;-1;1;5;1;-1;1;2;1;3|]

// In PageRank, only knowing the NUMBER of outgoing edges is enough to work with.
// The index is the node ID, the value is the number of edges outgoing from that node.
let outgoingEdgeSizes = [|1;2;3;4;0;2|]


// PageRank on the CPU.  This uses a 2D array for incoming edges and 1D array for number of outgoing edges.
let pagerankCPU iteration (ie:int [,]) (oes:int[]) = 
    let d = 0.85 // Damping factor
    let N : int = (ie.GetLength 0) // Number of nodes
    let Nf : float = float N // Number of nodes in float
    let damp : float = (1.0 / Nf)

    let rpr = Array.create N damp

    for rep in 0..iteration-1 do
        for i in 0..N-1 do
            let mutable sum = 0.0
            for j in 0..(ie.GetLength 1)-1 do
                if ie.[i,j] <> -1 then
                    let mutable size : float = float oes.[ie.[i,j]]
                    if size = 0.0 then
                        size <- 1.0
                    sum <- sum + rpr.[ie.[i,j]] / size
            // Apply damping factor
            rpr.[i] <- d * sum + (1.0 - d)/Nf
    
    printfn "%A" rpr
    0;;

[<EntryPoint>]
let main argv = 
    //let pr = pagerankCPU 100 incomingEdge outgoingEdgeSizes
    let prG = PageRank.pagerankGPU 100 2 6 incomingEdge1D outgoingEdgeSizes
    printfn "%A" prG    
    Console.ReadKey() |> ignore
    0