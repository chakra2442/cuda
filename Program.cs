using Alea;
using Alea.cuRAND;
using Alea.Parallel;
using NUnit.Framework;
using Samples.CSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace CUDATest
{

    class Program2
    {
        static void Main(string[] args)
        {
            Sim.Start();
            // PseudoRandomSamplingWithOffset();
            //var gpu = Gpu.Default;
            //gpu.Context.SetCurrent();
            //using (var window = new SimWindow(gpu, 0.001f))
            //{
            //    window.Run();
            //}
        }

        [GpuManaged, Test]
        public static void PseudoRandomSamplingWithOffset()
        {
            var numBlocks = 4;
            var numSamplesPerBlock = 1 << 16;
            using (var rng = Generator.CreateCpu(RngType.PSEUDO_MRG32K3A))
            {
                var gaussian = new double[numSamplesPerBlock];
                var poisson = new UInt32[numSamplesPerBlock];

                rng.SetPseudoRandomGeneratorSeed(42L);

                for (var block = 0; block < numBlocks; block++)
                {
                    rng.SetGeneratorOffset((ulong)block * (ulong)(numSamplesPerBlock));
                    rng.GenerateNormal(gaussian, 0, 1);
                    rng.GeneratePoisson(poisson, 1.0);

                    var sampleMeanGaussian = gaussian.Average();
                    Assert.That(sampleMeanGaussian, Is.EqualTo(0).Within(1e-2));

                    var sampleMeanPoisson = poisson.Select(p => (double)p).Average();
                    Assert.That(sampleMeanPoisson, Is.EqualTo(1.0).Within(1e-2));
                }
            }
        }
    }

    class Program
    {
        static void Main2(string[] args)
        {
            foreach (var device in Device.Devices)
            {
                Console.WriteLine($"{device.Id} : {device.Name} [{device.Attributes.ComputeCapabilityMajor}.{device.Attributes.ComputeCapabilityMinor}] {device.TotalMemory / 1024 / 1024}MB");
            }

            Console.WriteLine($"Default GPU: {Device.Default.Id}");

            try
            {
                var deviceId = 0;
                var numIters = 4;
                var numBatches = 8;
                var numSamplesPerBatch = (1 << 20) * 10;
                var showProgress = true;
                var doMultiGpuTest = false;

                //if (args.Length > 0)
                //{
                //    deviceId = Int32.Parse(args[0]);
                //}
                //else
                //{
                //    var deviceIds = Device.Devices.Select(device => device.Id);
                //    Console.Write("Select device id [");
                //    Console.Write(String.Join(", ", deviceIds.Select(id => id.ToString())));
                //    Console.Write("] : ");
                //    deviceId = Convert.ToInt32(Console.ReadLine());
                //}

                //if (args.Length > 1)
                //{
                //    doMultiGpuTest = Int32.Parse(args[1]) != 0;
                //}
                //else
                //{
                //    Console.Write("Do multi-GPU test? [0|1] : ");
                //    doMultiGpuTest = Convert.ToInt32(Console.ReadLine()) != 0;
                //}

                //if (args.Length > 2)
                //{
                //    numBatches = Int32.Parse(args[2]);
                //}
                //else
                //{
                //    Console.Write("Number of batches : ");
                //    numBatches = Convert.ToInt32(Console.ReadLine());
                //}

                //if (args.Length > 3)
                //{
                //    showProgress = Int32.Parse(args[3]) != 0;
                //}
                //else
                //{
                //    Console.Write("Show progress? [0|1] : ");
                //    showProgress = Convert.ToInt32(Console.ReadLine()) != 0;
                //}

                Console.WriteLine();

                var selected = Device.Devices.ToList().Find(device => device.Id == deviceId);
                if (showProgress) numIters = 1;
                FinanceAsianOptionMonteCarlo.ShowProgress = showProgress;
                FinanceAsianOptionMonteCarlo.DoMultiGpuTest = doMultiGpuTest;
                FinanceAsianOptionMonteCarlo.Run(selected, numIters, numBatches, numSamplesPerBatch);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"exception {ex.Message}");
            }
        }
    }
}
