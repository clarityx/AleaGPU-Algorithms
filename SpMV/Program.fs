(**
    Bryan Rowe
    UC Riverside
    "SpMV 1D Mapping"

    Entry point for SpMV GPU algorithm

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

// Data based on this matrix: http://op2.github.io/PyOP2/_images/csr.svg
let indices =   [|0;2;4;7;11;14|]
let values =    [|10;-2;3;9;7;8;7;3;8;7;5;8;9;13|]
let col =       [|0;4;0;1;1;2;3;0;2;3;4;1;3;4|]
let mulVector = [|1;2;3;4;5|]

[<EntryPoint>]
let main argv = 
    let resVector = Array.zeroCreate 5
    let nRows = 5
    let r = SpMV.SpMV nRows indices values col mulVector resVector

    printfn "%A" r

    Console.ReadKey() |> ignore
    0