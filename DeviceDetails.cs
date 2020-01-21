using Alea;
using Alea.CSharp;
using System;

namespace CUDATest
{
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
}
