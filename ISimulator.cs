using Alea;

namespace Samples.CSharp
{
    public interface ISimulator
    {
        string Description { get; }
        void RunNBodySim(deviceptr<float4> newPos,
                       deviceptr<float4> oldPos,
                       deviceptr<float4> vel,
                       deviceptr<float4> color,
                       int numBodies,
                       float deltaTime,
                       float softeningSquared,
                       float damping);
        float gborder { get; set; }
    }
}
