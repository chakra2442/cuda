using System;
using System.Linq;
using Alea;

namespace Samples.CSharp
{
    public abstract class BodyInitializer
    {
        abstract public int NumBodies { get; set; }
        abstract public float PScale { get; set; }
        abstract public float VScale { get; set; }
        public abstract float4 Position(int i);
        public abstract float4 Color(int i);
        public abstract float4 RColor(int i);
        public abstract float4 Velocity(float4 position, int i);

        static float4 Momentum(float4 velocity)
        {
            // we store mass in velocity.w
            var mass = velocity.w;
            return new float4(velocity.x * mass,
                              velocity.y * mass,
                              velocity.z * mass,
                              mass);
        }

        static public void Initialize(BodyInitializer initializer,
                                      float clusterScale,
                                      float velocityScale,
                                      int numBodies,
                                      out float4[] positions,
                                      out float4[] velocities,
                                      out float4[] colors)
        {
            var pscale = clusterScale * Math.Max(1.0f, numBodies / 1024.0f);
            var vscale = velocityScale * pscale;
            initializer.NumBodies = numBodies;
            initializer.PScale = pscale;
            initializer.VScale = vscale;
            positions = Enumerable.Range(0, numBodies).Select(initializer.Position).ToArray();
            colors = Enumerable.Range(0, numBodies).Select(initializer.Color).ToArray();

            velocities = positions.Select(initializer.Velocity).ToArray();

            // now we try to adjust velocity to make total momentum = zero.
            var momentums = velocities.Select(Momentum).ToArray();
            var totalMomentum = momentums.Aggregate(new float4(0.0f, 0.0f, 0.0f, 0.0f),
                (accum, momentum) =>
                    new float4(accum.x + momentum.x,
                               accum.y + momentum.y,
                               accum.z + momentum.z,
                               accum.w + momentum.w));
            Console.WriteLine("total momentum and mass 0 = {0}", totalMomentum);

            var len = velocities.Length;
            // adjust velocities
            velocities = velocities.Select((vel, i) => new float4(
                vel.x - totalMomentum.x / len / vel.w,
                vel.y - totalMomentum.y / len / vel.w,
                vel.z - totalMomentum.z / len / vel.w,
                vel.w)).ToArray();

            // see total momentum after adjustment
            momentums = velocities.Select(Momentum).ToArray();
            totalMomentum = momentums.Aggregate(new float4(0.0f, 0.0f, 0.0f, 0.0f),
                (accum, momentum) =>
                    new float4(accum.x + momentum.x,
                               accum.y + momentum.y,
                               accum.z + momentum.z,
                               accum.w + momentum.w));
            Console.WriteLine("total momentum and mass 1 = {0}", totalMomentum);
        }
    }
}
