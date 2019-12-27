using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Alea;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;

namespace CUDATest
{
    class TestWindow : GameWindow
    {
        private readonly Stopwatch _stopwatch;
        private int _frameCounter;
        private int _updateLag = 128;
        private int _vbo;

        public TestWindow(): base(1024, 768)
        {
            _stopwatch = Stopwatch.StartNew();

            var data = new[]
            {
                new Vector2(-1.0f, -1.0f),
                new Vector2(1.0f, -1.0f),
                new Vector2(1.0f, 1.0f),
                new Vector2(-1.0f, 1.0f),
                new Vector2(0.0f, 0.0f),
                new Vector2(1.0f, 0.0f),
                new Vector2(1.0f, 1.0f),
                new Vector2(0.0f, 1.0f),
            };

            GL.GenBuffers(1, out _vbo);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);
            GL.BufferData(BufferTarget.ArrayBuffer, (IntPtr)(Marshal.SizeOf(typeof(Vector2)) * data.Length), data, BufferUsageHint.DynamicDraw);
            GL.BindBuffer(BufferTarget.ArrayBuffer, 0);

        }


        private void Description()
        {
            var time = _stopwatch.ElapsedMilliseconds;
            var fps = ((float)_frameCounter) * 1000.0 / ((float)time);
            this.Title = $"fps {fps}";
            _stopwatch.Restart();
        }

        private void UpdateFPS()
        {
            _frameCounter++;
            if (_frameCounter >= _updateLag)
            {
                Description();
                _frameCounter = 0;
            }
        }

        protected override void OnRenderFrame(FrameEventArgs e)
        {
            //Color4 backColor;
            //backColor.A = 1.0f;
            //backColor.R = 0.1f;
            //backColor.G = 0.1f;
            //backColor.B = 0.3f;
            //GL.ClearColor(backColor);
            //GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            //var modelview = Matrix4.LookAt(Vector3.Zero, Vector3.UnitZ, Vector3.UnitY);
            //GL.MatrixMode(MatrixMode.Modelview);
            //GL.LoadMatrix(ref modelview);
            //GL.Color3(1.0f, 215.0f / 255.0f, 0.0f); // golden as the stars
            //GL.EnableClientState(ArrayCap.VertexArray);
            ////GL.BindBuffer(BufferTarget.ArrayBuffer, _buffers[1]);
            //GL.VertexPointer(4, VertexPointerType.Float, 0, 0);
            //GL.DrawArrays(PrimitiveType.Points, 0, 100);
            //GL.DisableClientState(ArrayCap.VertexArray);
            //GL.Finish();

            //SwapBuffers();


            //GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);
            //GL.VertexAttribPointer(posLocation, 2, VertexAttribPointerType.Float, false, 0, 0);
            //GL.EnableVertexAttribArray(posLocation);
            //GL.DrawArrays(BeginMode.Quads, positive ? 4 : 0, 4);
            //GL.DisableVertexAttribArray(posLocation);
            //GL.BindBuffer(BufferTarget.ArrayBuffer, 0);


        }

        protected override void OnUpdateFrame(FrameEventArgs e)
        {
            base.OnUpdateFrame(e);
            UpdateFPS();
        }

        protected override void OnResize(EventArgs e)
        {
            GL.Viewport(0, 0, Width, Height);
        }

        public void Dispose()
        {
            if (_vbo != 0)
            {
                GL.DeleteBuffers(1, ref _vbo);
                _vbo = 0;
            }
        }
    }
}
