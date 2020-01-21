using System;
using System.Linq;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Threading;
using Alea;
using Alea.Parallel;
using Alea.cuRAND;
using Alea.CSharp;
using NUnit.Framework;

namespace Samples.CSharp
{
    class FinanceAsianOptionMonteCarlo
    {
        private const ulong Seed = 42UL;
        const RngType RngType = Alea.cuRAND.RngType.PSEUDO_XORWOW;
        public static bool ShowProgress = false;
        public static bool DoMultiGpuTest = true;

        public static double AsianCall(int i, double spot0, double strike, double[] dt, double[] rates, double[] volas,
            double[] gaussian)
        {
            var numSamples = gaussian.Length/dt.Length;
            var sum = 0.0;
            var spot = spot0;
            for (var k = 0; k < dt.Length; k++)
            {
                var sigma = volas[k];
                var drift = dt[k]*(rates[k] - sigma*sigma/2);
                spot = spot*DeviceFunction.Exp(drift + DeviceFunction.Sqrt(dt[k])*sigma*gaussian[k*numSamples + i]);
                sum += spot;
            }
            return DeviceFunction.Max(sum/dt.Length - strike, 0.0);
        }

        public static double PriceCpu(int numBatches, int numSamplesPerBatch, double spot0, double strike, double[] dt,
            double[] rates, double[] volas)
        {
            using (var rng = Generator.CreateCpu(RngType))
            {
                rng.SetPseudoRandomGeneratorSeed(Seed);

                var nt = dt.Length;
                var gaussian = new double[numSamplesPerBatch*nt];
                var prices = new double[numSamplesPerBatch];
                var batchPrices = new double[numBatches];

                for (var batch = 0; batch < numBatches; batch++)
                {
                    rng.SetGeneratorOffset((ulong) batch*(ulong) (numSamplesPerBatch*nt));
                    rng.GenerateNormal(gaussian, 0, 1);

                    Parallel.For(0, numSamplesPerBatch,
                        i => prices[i] = AsianCall(i, spot0, strike, dt, rates, volas, gaussian));

                    batchPrices[batch] = prices.Sum()/numSamplesPerBatch;
                    if (ShowProgress) Console.Write("o");
                }

                if (ShowProgress) Console.WriteLine();

                return batchPrices.Sum()/numBatches;
            }
        }

        [GpuManaged]
        public static double PriceImplicit(Gpu gpu, ulong startOffset, int numBatches, int numSamplesPerBatch, double spot0, double strike,
            double[] dt, double[] rates, double[] volas)
        {
            using (var rng = Generator.CreateGpu(gpu, RngType))
            {
                rng.SetPseudoRandomGeneratorSeed(Seed);

                var nt = dt.Length;
                var gaussian = gpu.Allocate<double>(numSamplesPerBatch*nt);
                var prices = gpu.Allocate<double>(numSamplesPerBatch);
                var batchPrices = new double[numBatches];

                for (var batch = 0; batch < numBatches; batch++)
                {
                    rng.SetGeneratorOffset((ulong) batch*(ulong) (numSamplesPerBatch*nt) + startOffset);
                    rng.GenerateNormal(gaussian, 0, 1);

                    gpu.For(0, numSamplesPerBatch,
                        i => prices[i] = AsianCall(i, spot0, strike, dt, rates, volas, gaussian));

                    batchPrices[batch] = gpu.Sum(prices)/numSamplesPerBatch;

                    if (ShowProgress) Console.Write("o");
                }

                Gpu.Free(gaussian);
                Gpu.Free(prices);

                if (ShowProgress) Console.WriteLine();

                return batchPrices.Sum()/numBatches;
            }
        }

        public static double PriceExplicit(Gpu gpu, ulong startOffset, int numBatches, int numSamplesPerBatch, double spot0, double strike,
            double[] dt, double[] rates, double[] volas)
        {
            using (var rng = Generator.CreateGpu(gpu, RngType))
            {
                rng.SetPseudoRandomGeneratorSeed(Seed);

                var nt = dt.Length;
                var gaussian = gpu.Allocate<double>(numSamplesPerBatch*nt);
                var prices = gpu.Allocate<double>(numSamplesPerBatch);
                var dDt = gpu.Allocate(dt);
                var dRates = gpu.Allocate(rates);
                var dVolas = gpu.Allocate(volas);
                var batchPrices = new double[numBatches];

                for (var batch = 0; batch < numBatches; batch++)
                {
                    rng.SetGeneratorOffset((ulong) batch*(ulong) (numSamplesPerBatch*nt) + startOffset);
                    rng.GenerateNormal(gaussian, 0, 1);

                    gpu.For(0, numSamplesPerBatch,
                        i => prices[i] = AsianCall(i, spot0, strike, dDt, dRates, dVolas, gaussian));

                    batchPrices[batch] = gpu.Sum(prices)/numSamplesPerBatch;

                    if (ShowProgress) Console.Write("o");
                }

                Gpu.Free(gaussian);
                Gpu.Free(prices);
                Gpu.Free(dDt);
                Gpu.Free(dRates);
                Gpu.Free(dVolas);

                if (ShowProgress) Console.WriteLine();

                return batchPrices.Sum()/numBatches;
            }
        }

        [GpuManaged]
        public static double PriceTransformReduce(Gpu gpu, ulong startOffset, int numBatches, int numSamplesPerBatch, double spot0, double strike,
            double[] dt, double[] rates, double[] volas)
        {
            using (var rng = Generator.CreateGpu(gpu, RngType))
            {
                rng.SetPseudoRandomGeneratorSeed(Seed);

                var nt = dt.Length;
                var gaussian = gpu.Allocate<double>(numSamplesPerBatch*nt);
                var batchPrices = new double[numBatches];

                for (var batch = 0; batch < numBatches; batch++)
                {
                    rng.SetGeneratorOffset((ulong) batch*(ulong) (numSamplesPerBatch*nt) + startOffset);
                    rng.GenerateNormal(gaussian, 0, 1);

                    batchPrices[batch] = gpu.Aggregate(numSamplesPerBatch,
                        i => AsianCall(i, spot0, strike, dt, rates, volas, gaussian),
                        (a, b) => a + b) / numSamplesPerBatch;

                    if (ShowProgress) Console.Write("o");
                }

                Gpu.Free(gaussian);

                if (ShowProgress) Console.WriteLine();

                return batchPrices.Sum()/numBatches;
            }
        }

        public static double PriceConstantMemory(Gpu gpu, int numBatches, int numSamplesPerBatch, double spot0, double strike,
            GlobalArraySymbol<double> dt, GlobalArraySymbol<double> rates, GlobalArraySymbol<double> volas)
        {
            using (var rng = Generator.CreateGpu(gpu, RngType))
            {
                rng.SetPseudoRandomGeneratorSeed(Seed);

                var nt = dt.Length;
                var gaussian = gpu.Allocate<double>(numSamplesPerBatch*nt);
                var prices = gpu.Allocate<double>(numSamplesPerBatch);
                var batchPrices = new double[numBatches];

                for (var batch = 0; batch < numBatches; batch++)
                {
                    rng.SetGeneratorOffset((ulong) batch*(ulong) (numSamplesPerBatch*nt));
                    rng.GenerateNormal(gaussian, 0, 1);

                    gpu.For(0, numSamplesPerBatch, i =>
                    {
                        var sum = 0.0;
                        var spot = spot0;
                        for (var k = 0; k < nt; k++)
                        {
                            var sigma = volas[k];
                            var drift = dt[k]*(rates[k] - sigma*sigma/2);
                            spot = spot*DeviceFunction.Exp(drift + DeviceFunction.Sqrt(dt[k])*sigma*gaussian[k*numSamplesPerBatch + i]);
                            sum += spot;
                        }
                        prices[i] = DeviceFunction.Max(sum/nt - strike, 0.0);
                    });

                    batchPrices[batch] = gpu.Sum(prices)/numSamplesPerBatch;

                    if (ShowProgress) Console.Write("o");
                }

                Gpu.Free(gaussian);
                Gpu.Free(prices);

                if (ShowProgress) Console.WriteLine();

                return batchPrices.Sum()/numBatches;
            }
        }

        public static double PriceMultiGpu(int numTotalBatches, int numSamplesPerBatch, int nt, Func<Gpu, ulong, int, int, double> price)
        {
            if (numTotalBatches % Device.Devices.Length != 0)
            {
                throw new Exception("Batches cannot be divided");
            }

            var numBatches = numTotalBatches / Device.Devices.Length;
            var devicePrices = new double[Device.Devices.Length];

            var threads = Device.Devices.Select((device, i) =>
            {
                var gpu = Gpu.Get(device);
                var startOffset = (ulong)i * (ulong)(numSamplesPerBatch * nt) * (ulong)numBatches;
                return new Thread(() => devicePrices[i] = price(gpu, startOffset, numBatches, numSamplesPerBatch));
            }).ToArray();

            foreach (var thread in threads)
            {
                thread.Start();
            }

            foreach (var thread in threads)
            {
                thread.Join();
            }

            return devicePrices.Average();
        }

        public static double Time(Func<double> pricer, int numIter, string pricerName, bool warmUp = true)
        {
            var time = 0.0;
            var price = 0.0;

            // warmup - jit
            if (warmUp) pricer();

            // average time
            for (var i = 0; i < numIter; i++)
            {
                // free all implicit memory to make your performance more accurate
                Gpu.FreeAllImplicitMemory();
                var watch = Stopwatch.StartNew();
                price = pricer();
                watch.Stop();
                time += watch.Elapsed.TotalMilliseconds;
            }
            var avTime = time/numIter;
            Console.WriteLine($"price {price:F6}, average time {avTime:F2} ms\n");
            return avTime;
        }

        public static void Run(Device device, int numIter = 4, int numBatches = 8, int numSamplesPerBatch = (1 << 20) * 10)
        {
            var spot0 = 100.0;
            var strike = 110.0;
            var rates = new double[12] {0.01, 0.01, 0.015, 0.015, 0.02, 0.02, 0.02, 0.025, 0.025, 0.025, 0.025, 0.03};
            var volas = new double[12] {0.4, 0.35, 0.35, 0.3, 0.3, 0.3, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2};
            var times = Enumerable.Range(0, 13).Select(i => i*1.0/12.0).ToArray();
            var dt = times.Zip(times.Skip(1), (t0, t1) => t1 - t0).ToArray();
            var doMultiGpuTest = DoMultiGpuTest && !ShowProgress;

            var gpu = Gpu.Default;
            var mem = 8.0*numSamplesPerBatch*(dt.Length + 1)/(1 << 20);

            Console.WriteLine($"Asian option pricing benchmark [{gpu.Device.Name}]");
            Console.WriteLine($"Num iterations {numIter}, num batches {numBatches}, num samples per batch {numSamplesPerBatch}");
            Console.WriteLine($"Aprox GPU memory usage: {mem} MB");
            foreach (var theDevice in Device.Devices)
            {
                Console.WriteLine($"{theDevice.Id} : {theDevice.Name} [{theDevice.Attributes.ComputeCapabilityMajor}.{theDevice.Attributes.ComputeCapabilityMinor}] {theDevice.TotalMemory / 1024 / 1024}MB");
            }
            Console.WriteLine();

            // warm-up JIT compiler 
            foreach (var theDevice in Device.Devices)
            {
                if (doMultiGpuTest || theDevice.Id == device.Id)
                    Gpu.Get(theDevice).Launch(() => { }, new LaunchParam(1, 1));
            }

            Console.WriteLine("Gpu explicit memory management");
            var explicitMemMgt = Time(() => PriceExplicit(gpu, 0UL, numBatches, numSamplesPerBatch, spot0, strike, dt, rates, volas), numIter, "Gpu explicit memory management");

            if (doMultiGpuTest) Console.WriteLine("Gpu explicit memory management MultiGpu");
            var explicitMemMgtMultiGpu = (!doMultiGpuTest) ? 0.0 :
                Time(
                    () =>
                        PriceMultiGpu(numBatches, numSamplesPerBatch, dt.Length,
                            (theGpu, startOffset, theNumBatchs, theNumSamplesPerBatch) =>
                                PriceExplicit(theGpu, startOffset, theNumBatchs, theNumSamplesPerBatch, spot0, strike, dt,
                                    rates, volas)), numIter, "Gpu explicit memory management MultiGpu");

            Console.WriteLine("Gpu implicit memory management");
            var implicitMemMgt = Time(() => PriceImplicit(gpu, 0UL, numBatches, numSamplesPerBatch, spot0, strike, dt, rates, volas), numIter, "Gpu implicit memory management");

            if (doMultiGpuTest) Console.WriteLine("Gpu implicit memory management MultiGpu");
            var implicitMemMgtMultiGpu = (!doMultiGpuTest) ? 0.0 :
                Time(
                    () =>
                        PriceMultiGpu(numBatches, numSamplesPerBatch, dt.Length,
                            (theGpu, startOffset, theNumBatches, theNumSamplesPerBatch) =>
                                PriceImplicit(theGpu, startOffset, theNumBatches, theNumSamplesPerBatch, spot0, strike, dt,
                                    rates, volas)), numIter, "Gpu implicit memory management MultiGpu");

            Console.WriteLine("Gpu implicit memory management with transform-reduce");
            var implicitMemMgtWithTransfRed = Time(() => PriceTransformReduce(gpu, 0UL, numBatches, numSamplesPerBatch, spot0, strike, dt, rates, volas), numIter, "Gpu implicit memory management with transform-reduce");

            if (doMultiGpuTest) Console.WriteLine("Gpu implicit memory management with transform-reduce MultiGpu");
            var implicitMemMgtWithTransfRedMultiGpu = (!doMultiGpuTest) ? 0.0 :
                Time(
                    () =>
                        PriceMultiGpu(numBatches, numSamplesPerBatch, dt.Length,
                            (theGpu, startOffset, theNumBatches, theNumSamplesPerBatch) =>
                                PriceTransformReduce(theGpu, startOffset, theNumBatches, theNumSamplesPerBatch, spot0, strike, dt,
                                    rates, volas)), numIter, "Gpu implicit memory management with transform-reduce MultiGpu");

            Console.WriteLine("Gpu constant memory");
            var cDt = Gpu.DefineConstantArraySymbol<double>(dt.Length);
            var cRates = Gpu.DefineConstantArraySymbol<double>(rates.Length);
            var cVolas = Gpu.DefineConstantArraySymbol<double>(volas.Length);
            gpu.Copy(dt, cDt);
            gpu.Copy(rates, cRates);
            gpu.Copy(volas, cVolas);
            var constMem = Time(() => PriceConstantMemory(gpu, numBatches, numSamplesPerBatch, spot0, strike, cDt, cRates, cVolas), numIter, "Gpu constant memory");

            Console.WriteLine("Cpu");
            var cpu = Time(() => PriceCpu(numBatches, numSamplesPerBatch, spot0, strike, dt, rates, volas), 1, "Cpu");

            Console.WriteLine("Speedup");
            Console.WriteLine("=======");

            Console.WriteLine($"Gpu explicit memory management,                                {explicitMemMgt:F3} ms, {cpu / explicitMemMgt:F1}");
            if (doMultiGpuTest)
            Console.WriteLine($"Gpu explicit memory management MultiGpu,                       {explicitMemMgtMultiGpu:F3} ms, {cpu / explicitMemMgtMultiGpu:F1}");

            Console.WriteLine($"Gpu implicit memory management,                                {implicitMemMgt:F3} ms, {cpu / implicitMemMgt:F1}");
            if (doMultiGpuTest)
            Console.WriteLine($"Gpu implicit memory management MultiGpu,                       {implicitMemMgtMultiGpu:F3} ms, {cpu / implicitMemMgtMultiGpu:F1}");

            Console.WriteLine($"Gpu implicit memory management with transform-reduce,          {implicitMemMgtWithTransfRed:F3} ms, {cpu / implicitMemMgtWithTransfRed:F1}");
            if (doMultiGpuTest)
            Console.WriteLine($"Gpu implicit memory management with transform-reduce MultiGpu, {implicitMemMgtWithTransfRedMultiGpu:F3} ms, {cpu / implicitMemMgtWithTransfRedMultiGpu:F1}");

            Console.WriteLine($"Gpu constant memory,                                           {constMem:F3} ms, {cpu / constMem:F1}");
        }

        [Test]
        public static void RunOnDefault()
        {
            Run(Device.Default, numSamplesPerBatch: (1 << 20) * 2);
        }
    }
}
