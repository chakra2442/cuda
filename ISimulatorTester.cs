using Alea;

namespace Samples.CSharp
{
    public interface ISimulatorTester
    {
        string Description { get; }
        void Integrate(float4[] pos,
                       float4[] vel,
                       int numBodies,
                       float deltaTime,
                       float softeningSquared,
                       float damping,
                       int steps);
    }
}
