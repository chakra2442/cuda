using Alea;
using Alea.CSharp;
using Alea.cuRAND;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDATest
{
    public class Sim
    {
        public static int Count { get; internal set; }
        public static List<Agent> Population { get; internal set; }

        [GpuManaged]
        public static void Start()
        {
            Count = 100;
            Population = new List<Agent>();

            using (var rng = Generator.CreateGpu(Gpu.Default, RngType.PSEUDO_XORWOW))
            {
                rng.SetPseudoRandomGeneratorSeed(1023);
                var gaussian = new double[1 << 16];
                rng.GenerateNormal(gaussian, 1, 1);

                for (int i = 0; i < Count; i++)
                {
                    Population.Add(new Agent(i, gaussian));
                }
            }

            PrintState("Initial state");

            for (int i = 0; i < 1000000; i++)
            {
                //foreach (var agent in Population)
                //{
                //    agent.Work(i);
                //}

                // Parallel.ForEach(Population, (x) => x.Work(i));

                Gpu.Default.Launch(CUDAKernel, new LaunchParam(1024, 1024), Population.ToArray());
            }

            PrintState("Final state");
        }

        [GpuManaged]
        private static void CUDAKernel(Agent [] input)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;
            for (var i = start; i < Count; i += stride)
            {
                
                    input[i].Work(-1);
            }
        }

        private static void PrintState(string v)
        {
            var totalResources = Enumerable.Sum(Population, x => x.Resource);
            var totalSkills = Enumerable.Sum(Population, x => x.Skill);

            Console.WriteLine($"{v} Skills : {totalSkills}, Resources : {totalResources}, cI : {Agent.sCountInteraction}, cCop : {Agent.sCountCooperation}");

        }
    }
}
