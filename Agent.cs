using Alea;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CUDATest
{
    public class Agent
    {
        static public double pInteraction = 0.5;
        static public double pCooperation = 0.1;
        static public int sCountInteraction = 0;
        static public int sCountCooperation = 0;

        [GpuParam]
        private double[] rand;

        public int Id { get; set; }
        public double Skill { get; set; }
        public double Resource { get; set; }

        public Agent(int id, double[] gaussian)
        {
            Id = id;
            rand = gaussian;
            Skill = rand[id % rand.Length];
            Resource = rand[(id * id) % rand.Length];
        }

        [GpuManaged]
        public void Work(int n)
        {
            if (GetRand(n) <= pInteraction)
            {
                sCountInteraction++;

                var tId = GetRandInt(n) % Sim.Count;
                if (tId == Id)
                {
                    return;
                }

                Agent target = Sim.Population[tId];

                if (GetRand(n) <= pCooperation)
                {
                    sCountCooperation++;
                    var delta = target.Skill + this.Skill;
                    target.Resource += delta;
                    this.Resource += delta;
                }
                else
                {
                    this.Resource += target.Resource;
                    target.Resource = 0;
                }
            }
            else
            {
                Resource = Resource + Skill;
            }
        }

        public double GetRand(int n)
        {
            return rand[((n + 1) * Id) % rand.Length];
        }

        public int GetRandInt(int n)
        {
            var x = rand[((n + 1) * Id) % rand.Length];
            return (int)(rand.Length * x * x);
        }
    }
}
