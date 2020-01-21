using Alea;
using Alea.CSharp;
using System;
using System.Diagnostics;
using System.Linq;

namespace CUDATest
{
    class BenchMark
    {
        private static int Length = 1000000;

        public static void Run()
        {
            var lp = LaunchParam.AutoByBlockSize(100);
            for (int k = 0; k < 3; k++)
            {
                var watch = Stopwatch.StartNew();
                var result = Enumerable.Repeat(0, Length).ToArray();
                var tMem = watch.Elapsed;
                Gpu.Default.Launch(RKernel, new LaunchParam(1024, 1024), result);
                var tGpu = watch.Elapsed;
                Console.WriteLine($"{Length} : MEM :{tMem.TotalMilliseconds}, GPU : {tGpu.TotalMilliseconds}, Avg : {tGpu.TotalMilliseconds/Length}");
                Length *= 10;
            }
            for (int k = 0; k < 3; k++)
            {
                Length /= 10;
                var watch = Stopwatch.StartNew();
                var result = Enumerable.Repeat(0, Length).ToArray();
                var tMem = watch.Elapsed;
                Gpu.Default.Launch(RKernel, new LaunchParam(1024, 1024), result);
                var tGpu = watch.Elapsed;
                Console.WriteLine($"{Length} : MEM :{tMem.TotalMilliseconds}, GPU : {tGpu.TotalMilliseconds}, Avg : {tGpu.TotalMilliseconds / Length}");
            }

            
        }

        private static void RKernel(int[] result)
        {
            //var r = new Random();
            //int res = r.Next() * r.Next();
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < result.Length; i += stride)
            {
                result[i] = -1;
                var x = i;
                for (int j = 0; j < 1000; j++)
                {
                    x *= x;
                }
                result[i] = (int) x;
            }
        }

        private static void Kernel(int[] result, int[] arg1, int[] arg2)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < result.Length; i += stride)
            {
                result[i] = arg1[i] + arg2[i];
            }
        }
    }
}
