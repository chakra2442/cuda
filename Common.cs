using System;
using Alea;
using NUnit.Framework;

namespace Samples.CSharp
{
    public static class Common
    {
        public static int DivUp(int num, int den)
        {
            return (num + den - 1)/den;
        }

        //public static void Test(
        //    BodyInitializer initializer,
        //    ISimulatorTester expectedSimulator,
        //    ISimulatorTester actualSimulator,
        //    int numBodies)
        //{
        //    const float clusterScale = 1.0f;
        //    const float velocityScale = 1.0f;
        //    const float deltaTime = 0.001f;
        //    const float softeningSquared = 0.00125f;
        //    const float damping = 0.9995f;
        //    const int steps = 5;

        //    Console.WriteLine("Testing {0} against {1} with {2} bodies...",
        //        actualSimulator.Description,
        //        expectedSimulator.Description,
        //        numBodies);
        //    Console.WriteLine("Using body initializer {0}...", initializer);

        //    float4[] expectedPos, expectedVel;
        //    BodyInitializer.Initialize(initializer, clusterScale, velocityScale, numBodies,
        //                               out expectedPos, out expectedVel);

        //    for (var i = 0; i < steps; i++)
        //    {
        //        const double tol = 1e-5;
        //        var actualPos = new float4[numBodies];
        //        var actualVel = new float4[numBodies];
        //        Array.Copy(expectedPos, actualPos, numBodies);
        //        Array.Copy(expectedVel, actualVel, numBodies);
        //        expectedSimulator.Integrate(expectedPos, expectedVel, numBodies, deltaTime,
        //                                    softeningSquared, damping, 1);
        //        actualSimulator.Integrate(actualPos, actualVel, numBodies, deltaTime,
        //                                  softeningSquared, damping, 1);
        //        for (var j = 0; j < expectedPos.Length; j++)
        //        {
        //            Assert.AreEqual(actualPos[j].x, expectedPos[j].x, tol);
        //            Assert.AreEqual(actualPos[j].y, expectedPos[j].y, tol);
        //            Assert.AreEqual(actualPos[j].z, expectedPos[j].z, tol);
        //            Assert.AreEqual(actualPos[j].w, expectedPos[j].w, tol);
        //        }
        //    }
        //}

        //public static void Test(ISimulatorTester expectedSimulator, ISimulatorTester actualSimulator,
        //                        int numBodies)
        //{
        //    Test(new BodyInitializerImpl(), expectedSimulator, actualSimulator, numBodies);
        //}

        //public static void Performance(ISimulatorTester simulator, int numBodies)
        //{
        //    const float clusterScale = 1.0f;
        //    const float velocityScale = 1.0f;
        //    const float deltaTime = 0.001f;
        //    const float softeningSquared = 0.00125f;
        //    const float damping = 0.9995f;
        //    const int steps = 10;

        //    Console.WriteLine("Perfomancing {0} with {1} bodies...", simulator.Description, numBodies);

        //    float4[] pos, vel;
        //    BodyInitializer.Initialize(new BodyInitializerImpl(), clusterScale, velocityScale, numBodies,
        //                               out pos, out vel);
        //    simulator.Integrate(pos, vel, numBodies, deltaTime, softeningSquared, damping, steps);
        //}


        public static float3 BodyBodyInteraction(float softeningSquared, float3 ai, float4 bi, float4 bj, float gborder = 0.0f)
        {
            // r_ij  [3 FLOPS]
            var r = new float3(bj.x - bi.x, bj.y - bi.y, bj.z - bi.z);

            // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
            var distSqr = r.x * r.x + r.y * r.y + r.z * r.z + softeningSquared;

            

            // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
            var invDist = LibDevice.__nv_rsqrtf(distSqr);
            var invDistCube = invDist * invDist * invDist ;

            //if (distSqr < gborder)
            //{
            //    invDistCube = gborder* gborder;
            //}
            //else
            //{
            //    invDistCube = gborder;
            //}


            // s = m_j * invDistCube [1 FLOP]
            var s = bj.w * invDistCube;
            // var s = gborder;

            // a_i =  a_i + s * r_ij [6 FLOPS]
            return (new float3(ai.x + r.x * s, ai.y + r.y * s, ai.z + r.z * s));
        }
    }
}
