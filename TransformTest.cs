using Alea;
using Alea.CSharp;
using NUnit.Framework;
using System.Linq;

namespace CUDATest
{
    class TransformTest
    {
        private const int Length = 1000000;

        private static void Kernel(int[] result, int[] arg1, int[] arg2)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < result.Length; i += stride)
            {
                result[i] = arg1[i] + arg2[i];
            }
        }

        [GpuManaged, Test]
        public static void Run()
        {
            var gpu = Gpu.Default;
            var lp = new LaunchParam(16, 256);
            var arg1 = Enumerable.Range(0, Length).ToArray();
            var arg2 = Enumerable.Range(0, Length).ToArray();
            var result = new int[Length];
            gpu.Launch(Kernel, lp, result, arg1, arg2);
        }
    }
}
