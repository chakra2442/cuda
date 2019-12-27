using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using Alea;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;

namespace Samples.CSharp
{
    public class SimWindow : GameWindow
    {
        private readonly int _numBodies;
        private readonly float _deltaTime;
        private readonly float _softeningSquared;
        private readonly float _damping;
        private readonly Gpu _gpu;
        private ISimulator _simulator;
        private readonly uint[] _buffers;
        private readonly DeviceMemory<float4> _vel;

        //private readonly DeviceMemory<int3> _color;
        //private readonly uint colorBuffer;

        private readonly IntPtr[] _resources;
        private readonly Stopwatch _stopwatch;
        private readonly int _fpsCalcLag;
        private int _frameCounter;

        void Description()
        {
            var time = _stopwatch.ElapsedMilliseconds;
            var fps = ((float)_frameCounter) * 1000.0 / ((float)time);
            Title = $"bodies {_numBodies}, {_gpu.Device.Name} {_gpu.Device.Arch} {_gpu.Device.Cores} cores, {_simulator.Description}, fps {fps}";
            _stopwatch.Restart();
        }

        void LockPos(Action<deviceptr<float4>, deviceptr<float4>> f)
        {
            CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsResourceSetMapFlags(_resources[0],
                (uint)CUDAInterop.CUgraphicsMapResourceFlags_enum.CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY));
            CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsResourceSetMapFlags(_resources[1],
                (uint)CUDAInterop.CUgraphicsMapResourceFlags_enum.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD));
            CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsMapResourcesEx(2u, _resources, IntPtr.Zero));

            var bytes = IntPtr.Zero;
            var handle0 = IntPtr.Zero;
            var handle1 = IntPtr.Zero;
            unsafe
            {
                CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsResourceGetMappedPointer(&handle0, &bytes,
                                                                                      _resources[0]));
                CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsResourceGetMappedPointer(&handle1, &bytes,
                                                                                      _resources[1]));
            }
            var pos0 = new deviceptr<float4>(handle0);
            var pos1 = new deviceptr<float4>(handle1);
            try
            {
                f(pos1, pos0);
            }
            finally
            {
                CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsUnmapResourcesEx(2u, _resources, IntPtr.Zero));
            }
        }

        public SimWindow(Gpu gpu, float gborder) : base(800, 600, GraphicsMode.Default, "Gravitational n-body simulation")
        {
            this.X = 0;
            this.Y = 0;

            _numBodies =  256*64;
            //_numBodies = 256 * 3; // CPU
            const float clusterScale = 1.0f;
            const float velocityScale = 1.0f;
            _deltaTime = 0.001f;
            _softeningSquared = 0.005f;
            _damping = 0.99999f;
            _gpu = gpu;

            _stopwatch = Stopwatch.StartNew();
            _fpsCalcLag = 128;
            _frameCounter = 0;

            _simulator = new GpuSimulator(_gpu, 256);
            _simulator.gborder = gborder;
            _buffers = new uint[4];
            for (var i = 0; i < _buffers.Length; i++)
            {
                _buffers[i] = 0;
            }
            GL.GenBuffers(_buffers.Length, _buffers);
            foreach (var buffer in _buffers)
            {
                GL.BindBuffer(BufferTarget.ArrayBuffer, buffer);
                GL.BufferData(BufferTarget.ArrayBuffer, (IntPtr)(Gpu.SizeOf<float4>() * _numBodies), IntPtr.Zero, BufferUsageHint.DynamicDraw);
                var size = 0;
                unsafe
                {
                    GL.GetBufferParameter(BufferTarget.ArrayBuffer, BufferParameterName.BufferSize, &size);
                }
                if (size != Gpu.SizeOf<float4>() * _numBodies)
                {
                    throw new Exception("Pixel Buffer Object allocation failed!");
                }
                GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
                CUDAInterop.cuSafeCall(CUDAInterop.cuGLRegisterBufferObject(buffer));
            }

            _resources = new IntPtr[_buffers.Length];

            for (var i = 0; i < _buffers.Length; i++)
            {
                var res = IntPtr.Zero;
                unsafe
                {
                    CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsGLRegisterBuffer(&res, _buffers[i], 0u));
                }
                _resources[i] = res;
            }

            _vel = _gpu.AllocateDevice<float4>(_numBodies);

            float4[] hpos, hvel, hcol;
            BodyInitializer.Initialize(new BodyInitializerImpl(), clusterScale, velocityScale, _numBodies, out hpos, out hvel, out hcol);
            Gpu.Copy(hvel, 0L, _gpu, _vel.Ptr, hvel.Length);



            // _color = _gpu.AllocateDevice<int3>(_numBodies);

            GL.BindBuffer(BufferTarget.ArrayBuffer, _buffers[2]);
            GL.BufferData(BufferTarget.ArrayBuffer, (IntPtr)(Marshal.SizeOf(typeof(float4)) * _numBodies), hcol, BufferUsageHint.DynamicDraw);
            // Gpu.Copy(colors, 0L, _gpu, _color.Ptr, _numBodies);

            LockPos((pos0, pos1) => Gpu.Copy(hpos, 0L, _gpu, pos1, hpos.Length));
            Description();
        }


        public void SwapPos()
        {
            var buffer = _buffers[0];
            _buffers[0] = _buffers[1];
            _buffers[1] = buffer;

            var resource = _resources[0];
            _resources[0] = _resources[1];
            _resources[1] = resource;
        }

        void HandleKeyDown(object sender, KeyboardKeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.Escape:
                    Exit();
                    break;
                case Key.S:
                    // SwitchSimulator();
                    break;
            }
        }

        protected override void Dispose(bool disposing)
        {
            try
            {
                foreach (var resource in _resources)
                {
                    CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsUnregisterResource(resource));
                }
                foreach (var buffer in _buffers)
                {
                    CUDAInterop.cuSafeCall(CUDAInterop.cuGLUnregisterBufferObject(buffer));
                }
                if (_buffers.Length > 0)
                {
                    GL.DeleteBuffers(_buffers.Length, _buffers);
                }
                if (disposing)
                {
                    _vel.Dispose();
                }
            }
            catch { }
            base.Dispose(disposing);
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            base.KeyDown += HandleKeyDown;
            VSync = VSyncMode.Off;
            GL.ClearColor(0.0f, 0.0f, 0.0f, 0.0f); // black as the universe
            GL.Enable(EnableCap.DepthTest);
        }

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);
            GL.Viewport(ClientRectangle.X, ClientRectangle.Y, ClientRectangle.Width, ClientRectangle.Height);
            var projection = Matrix4.CreatePerspectiveFieldOfView((float)Math.PI / 4.0f,
                                                                  (float)Width / Height, 1.0f, 64.0f);
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadMatrix(ref projection);
        }

        protected override void OnRenderFrame(FrameEventArgs e)
        {
            base.OnRenderFrame(e);

            _frameCounter++;
            if (_frameCounter >= _fpsCalcLag)
            {
                Description();
                _frameCounter = 0;
            }

            SwapPos();
            LockPos((pos0, pos1) =>
                _simulator.RunNBodySim(pos1, pos0, _vel.Ptr, _numBodies, _deltaTime, _softeningSquared, _damping));

            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            var modelview = Matrix4.LookAt(Vector3.Zero, Vector3.UnitZ, Vector3.UnitY);
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadMatrix(ref modelview);

            // GL.Color3(1.0f, 215.0f / 255.0f, 0.0f); // golden as the stars
            GL.EnableClientState(ArrayCap.VertexArray);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _buffers[1]);
            GL.VertexPointer(4, VertexPointerType.Float, 0, 0);

            GL.EnableClientState(ArrayCap.ColorArray);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _buffers[2]);
            GL.ColorPointer(4, ColorPointerType.Float, 0, 0);

            GL.DrawArrays(PrimitiveType.Points, 0, _numBodies);
            GL.DisableClientState(ArrayCap.VertexArray);

            GL.Finish();
            SwapBuffers();
        }
    }
}
