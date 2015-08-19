(**
    Bryan Rowe
    UC Riverside
    "ALeaGPU PageRank"

    This file contains code to run PageRank on the GPU

    Version 1.0
    todo: optimize iterations
**)
module PageRank

#if SCRIPT_REFS
#r "..\\..\\..\\packages\\Alea.IL\\lib\\net40\\Alea.IL.dll"
#r "..\\..\\..\\packages\\Alea.CUDA\\lib\\net40\\Alea.CUDA.dll"
#r "..\\..\\..\\packages\\Alea.CUDA.IL\\lib\\net40\\Alea.CUDA.IL.dll"
#r "..\\..\\..\\packages\\NUnit\\lib\\nunit.framework.dll"
#endif

open FSharp.Charting
open System
open System.IO
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.IL
open NUnit.Framework

let worker = Worker.Default
let NODE_SIZE = 6



(*** define:pagerankKernel ***)  
[<ReflectedDefinition>]
let pagerankKernel (w:int) (h:int) (ie:deviceptr<int>) (oes:deviceptr<int>) (rpr:deviceptr<float>) =
    let id = blockIdx.x * blockDim.x + threadIdx.x // Unique thread ID
    let d = 0.85 // Damping factor
    let N : int = h // Number of nodes
    let Nf : float = float N // Number of nodes in float
    let damp : float = (1.0 / Nf)

    // Function: Treating a 1D array as 2D array
    let getVal x y = ie.[w*x+y]

    if id >= 0 && id < N then
        let mutable sum = 0.0
        for j in 0..w-1 do
            let mutable value = getVal id j
            if value <> -1 then
                let mutable size : float = float oes.[value]
                if size = 0.0 then
                    size <- 1.0
                sum <- sum + rpr.[value] / size
        __syncthreads()
        rpr.[id] <- d * sum + (1.0 - d)/Nf       

let pagerankGPU_inner w h (ie:int[]) (oes:int[]) (rpr:float[]) =    
    use ie_inputs = worker.MallocArray(ie)
    use oes_inputs = worker.MallocArray(oes)
    use rpr_outputs = worker.MallocArray(rpr) // Running PageRank array to store results
    let blockSize = 256
    let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
    let gridSize = Math.Min(16 * numSm, divup 10000 blockSize)
    let lp = new LaunchParam(gridSize, blockSize)
    worker.Launch <@ pagerankKernel @> lp w h ie_inputs.Ptr oes_inputs.Ptr rpr_outputs.Ptr
    rpr_outputs.Gather()

let pagerankGPU iteration w h (ie:int[]) (oes:int[]) =
    let Nf : float = float h
    let damp : float = (1.0 / Nf)

    let mutable rpr = Array.create<float> h damp

    if iteration > 1 then
        for i in 0..iteration-1 do
            let pr = pagerankGPU_inner w h ie oes rpr
            rpr <- pr
    rpr