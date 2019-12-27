using System;
using Alea;

namespace Samples.CSharp
{
    public class BodyInitializerImpl : BodyInitializer
    {
        private readonly Random _random = new Random();

        public override int NumBodies { get; set; }
        public override float PScale { get; set; }
        public override float VScale { get; set; }

        private float Rand(float scale, float location)
        {
            return (float)(_random.NextDouble() * scale + location);
        }

        private float RandP()
        {
            return PScale * Rand(1.0f, -0.5f);
        }

        private float RandV()
        {
            return VScale * Rand(1.0f, -0.5f);
        }

        private float RandM()
        {
            return Rand(0.6f, 0.7f);
        }

        public override float4 Position(int i)
        {
            if (i < NumBodies / 2)
            {
                return new float4(RandP() + 0.5f * PScale, RandP(), RandP() + 50.0f, RandM());
            }
            else
            {
                return new float4(RandP() - 0.5f * PScale, RandP(), RandP() + 50.0f, RandM());
            }
        }

        public override float4 Velocity(float4 position, int i)
        {
            if (i < NumBodies / 2)
            {
                return new float4(RandV(), RandV() + 0.01f * VScale * position.x * position.x, RandV(), position.w);
            }
            else
            {
                return new float4(RandV(), RandV() - 0.01f * VScale * position.x * position.x, RandV(), position.w);
            }
        }

        public override float4 Color(int i)
        {
            var r = (float)(_random.NextDouble());
            var g = (float)(_random.NextDouble());
            var b = (float)(_random.NextDouble());
            var w = (float)(_random.NextDouble());
            return new float4(r, g, b, w);
        }
    }
}
