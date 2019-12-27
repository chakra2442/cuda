using Alea;
using Alea.CSharp;
using Alea.Parallel;
using NUnit.Framework;
using Samples.CSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace CUDATest
{

    class Program
    {
        static void Main(string[] args)
        {
            //DeviceDetails.Print();

            //DeviceDetails.PrintGridBlock();

            // ParallelForTest.GPUParallelFor();
            // ParallelForChainingTest.ChainLambdas();

            //for (int i = 0; i < 1000; i++)
            //{
            //    TransformTest.Run();
            //    Console.WriteLine(i);
            //}


            var gb = 0.1f;
            
            if (args.Length > 0)
            {
                float.TryParse(args[0], out gb);
                Console.WriteLine(gb);
            }

            var list = new List<Task>();
            list.Add(Task.Run(() =>
            {
                var gpu = Gpu.Default;
                gpu.Context.SetCurrent();
                using (var window = new SimWindow(gpu, gb))
                {
                    window.Run();
                }
            }));

            //list.Add(Task.Run(() =>
            //{
            //    var gpu = Gpu.Default;
            //    gpu.Context.SetCurrent();
            //    using (var window = new SimWindow(gpu, 0.001f))
            //    {
            //        window.Run();
            //    }
            //}));

            Task.WaitAll(list.ToArray());

            //using (var tw = new TestWindow())
            //{
            //    tw.Run(150.0);
            //}

        }
    }

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

    class DeviceDetails
    {
        public static void Print()
        {
            var devices = Device.Devices;
            var numGpus = devices.Length;
            foreach (var device in devices)
            {
                device.Print();

                // note that device ids for all GPU devices in a system does not need to be continuous
                var id = device.Id;
                var arch = device.Arch;
                var numMultiProc = device.Attributes.MultiprocessorCount;
            }
        }

        public static void PrintGridBlock()
        {
            Gpu.Default.Launch(() => Console.WriteLine("block index {0}, thread index {1}", blockIdx.x, threadIdx.x), new LaunchParam(16, 64));
            Console.Write("Press Enter to quit...");
            Console.ReadKey();
        }
    }

    class ParallelForTest
    {
        private const int Length = 1000000;

        public static void GPUParallelFor()
        {
            var arg1 = Enumerable.Range(0, Length).ToArray();
            var arg2 = Enumerable.Range(0, Length).ToArray();
            var result = new int[Length];

            var watch = Stopwatch.StartNew();
            Parallel.For(0, result.Length, i => result[i] = arg1[i] + arg2[i]);
            var tCpu = watch.Elapsed;

            watch.Restart();
            Gpu.Default.For(0, result.Length, i => result[i] = arg1[i] + arg2[i]);
            var tGpu = watch.Elapsed;

            watch.Restart();
            Action<int> op = i => result[i] = arg1[i] + arg2[i];
            Gpu.Default.For(0, arg1.Length, op);
            var tGpuA = watch.Elapsed;

            watch.Restart();
            Func<int[], Action<int>> opFactory = res => i => res[i] = arg1[i] + arg2[i];
            Gpu.Default.For(0, arg1.Length, opFactory(result));
            var tGpuF = watch.Elapsed;

            Console.WriteLine($"{Length} : CPU :{tCpu.TotalMilliseconds}, GPU : {tGpu.TotalMilliseconds}, GPUA : {tGpuA.TotalMilliseconds}, GPUF : {tGpuF.TotalMilliseconds}");
        }
    }

    class ParallelForChainingTest
    {
        private const int Length = 1000000;

        public static void ChainLambdas()
        {
            var gpu = Gpu.Default;
            var arg1 = Enumerable.Range(0, Length).ToArray();
            var arg2 = Enumerable.Range(0, Length).ToArray();
            var arg3 = Enumerable.Range(0, Length).ToArray();
            var result = new int[Length];

            var watch = Stopwatch.StartNew();
            Parallel.For(0, result.Length, i => result[i] = (arg1[i] + arg2[i])* arg3[i] );
            var tCpu = watch.Elapsed;

            watch.Restart();
            gpu.For(0, result.Length, i => result[i] = arg1[i] + arg2[i]);
            gpu.For(0, result.Length, i => result[i] = result[i] * arg3[i]);
            var tGpu = watch.Elapsed;
            Console.WriteLine($"{Length} : CPU :{tCpu.TotalMilliseconds}, GPU : {tGpu.TotalMilliseconds}");
        }
    }
}
