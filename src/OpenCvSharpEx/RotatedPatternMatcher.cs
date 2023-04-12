using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp;
using OpenCvSharpEx.Internal;

namespace OpenCvSharpEx
{
    public class RotatedPattenMatcher : IDisposable
    {
        public RotatedPattenMatcher()
        {

        }
        IntPtr rotatedPatternMatcherObj;
        public double AcceptancePercentage
        {
            get;
            set;
        } = 90.0;
        public double MinAngle
        {
            get;
            set;
        }
        public double MaxAngle
        {
            get;
            set;
        }
        public double AngleStep
        {
            get;
            set;
        } = 1.0;
        public int MinReducedArea
        {
            get;
            set;
        } = 1000;

        public void Teach(Mat pattern)
        {
            var ret = NativeMethods.rotatedPatternMatcher_RotatedPatternMatcher_new(pattern.CvPtr, this.MinAngle, this.MaxAngle, this.AngleStep, this.MinReducedArea, out this.rotatedPatternMatcherObj);
        }
        public void PreprocessPattern()
        {

        }
        public RotationPatternMatcherResults Search(Mat image, bool refineResults = false)
        {
            if (this.rotatedPatternMatcherObj == IntPtr.Zero)
                throw new OpenCvSharpException("No pattern is taught yet.");
            double score = this.AcceptancePercentage;
            var ret = NativeMethods.rotatedPatternMatcher_RotatedPatternMatcher_search(this.rotatedPatternMatcherObj, image.CvPtr, out var location, out var angle, ref score);
            return new RotationPatternMatcherResults()
            {
                Location = location,
                Angle = angle,
                Score = score,
            };
        }
        ~RotatedPattenMatcher()
        {
            this.Dispose();
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            if (this.rotatedPatternMatcherObj != IntPtr.Zero)
            {
                NativeMethods.rotatedPatternMatcher_RotatedPatternMatcher_delete(this.rotatedPatternMatcherObj);
                this.rotatedPatternMatcherObj = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }
    }
    public class RotationPatternMatcherResults
    {
        public Point2d Location { get; set; }
        public double Angle { get; set; }
        public RotatedRect RotatedBounds { get; set; }
        public Rect2d Bounds { get; set; }
        public double Score { get; set; }
    }
}
