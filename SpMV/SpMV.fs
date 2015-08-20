(**
    Bryan Rowe
    UC Riverside
    "SpMV 1D Mapping"

    Run SpMV on the GPU

    Version 1.0
**)

module SpMV

#if SCRIPT_REFS
#r "..\\..\\..\\packages\\Alea.IL\\lib\\net40\\Alea.IL.dll"
#r "..\\..\\..\\packages\\Alea.CUDA\\lib\\net40\\Alea.CUDA.dll"
#r "..\\..\\..\\packages\\Alea.CUDA.IL\\lib\\net40\\Alea.CUDA.IL.dll"
#r "..\\..\\..\\packages\\NUnit\\lib\\nunit.framework.dll"
#endif

open System
open System.IO
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.IL
open NUnit.Framework

let worker = Worker.Default

(*** define:pagerankKernel ***)  
[<ReflectedDefinition>]
let SpMV_kernel_1d_mapping (nRows:int) (indices:deviceptr<int>) (values:deviceptr<int>) (col:deviceptr<int>) (mulVector:deviceptr<int>) (resVector:deviceptr<int>)  =
    let globalID = blockIdx.x * blockDim.x + threadIdx.x

    if globalID >= 0 && globalID < nRows then
        let mutable sum = 0
        let startingPosition = indices.[globalID]
        let endingPosition   = indices.[globalID + 1]
        for i in startingPosition..endingPosition-1 do
            let mutable tmp = values.[i] * mulVector.[col.[i]]
            sum <- sum + tmp
        resVector.[globalID] <- sum

let SpMV (nRows:int) (indices:int[]) (values:int[]) (col:int[]) (mulVector:int[]) (resVector:int[]) =    
    use indices_inputs =    worker.MallocArray(indices)
    use values_inputs =     worker.MallocArray(values)
    use col_inputs =        worker.MallocArray(col)
    use mulVector_inputs =  worker.MallocArray(mulVector)
    use resVector_outputs = worker.MallocArray(resVector)
    let blockSize = 256
    let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
    let gridSize = Math.Min(16 * numSm, divup values.Length blockSize)
    let lp = new LaunchParam(gridSize, blockSize)
    worker.Launch <@ SpMV_kernel_1d_mapping @> lp nRows indices_inputs.Ptr values_inputs.Ptr col_inputs.Ptr mulVector_inputs.Ptr resVector_outputs.Ptr
    resVector_outputs.Gather()