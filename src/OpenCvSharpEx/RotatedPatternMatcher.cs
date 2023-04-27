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
        } = -180.0;
        public double MaxAngle
        {
            get;
            set;
        } = 180.0;
        public double AngleStep
        {
            get;
            set;
        } = 1.0;
        public int MinReducedArea
        {
            get;
            set;
        } = 256;

        public int MaxMatchCount
        {
            get;
            set;
        } = 10;
        public double MaxOverlapRatio
        {
            get;
            set;
        } = 0.0;
        public void Teach(Mat pattern)
        {
            var ret = NativeMethods.rotatedPatternMatcher_RotatedPatternMatcher_new(pattern.CvPtr, this.MinAngle, this.MaxAngle, this.AngleStep, this.MinReducedArea, out this.rotatedPatternMatcherObj);
        }
        public void PreprocessPattern()
        {

        }
        public RotationPatternMatcherResults[] Search(Mat image, bool refineResults = false)
        {
            if (this.rotatedPatternMatcherObj == IntPtr.Zero)
                throw new OpenCvSharpException("No pattern is taught yet.");
            double score = this.AcceptancePercentage;
            var ret = NativeMethods.rotatedPatternMatcher_RotatedPatternMatcher_search(this.rotatedPatternMatcherObj, image.CvPtr, this.AcceptancePercentage, this.MinAngle, this.MaxAngle, this.AngleStep, this.MaxMatchCount, this.MinReducedArea, this.MaxOverlapRatio, out var results, out var resultsLength);
            var r = new RotationPatternMatcherResults[resultsLength];
            var p = results;
            for (int i = 0; i < r.Length; ++i)
            {
                r[i] = (RotationPatternMatcherResults)System.Runtime.InteropServices.Marshal.PtrToStructure(p, typeof(RotationPatternMatcherResults));
                p += System.Runtime.InteropServices.Marshal.SizeOf(typeof(RotationPatternMatcherResults));
            }
            System.Runtime.InteropServices.Marshal.FreeCoTaskMem(results);
            return r;
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
    public struct RotationPatternMatcherResults
    {
        public Point2d Location { get; set; }
        public double Angle { get; set; }
        public RotatedRect RotatedBounds { get; set; }
        public Rect2d Bounds { get; set; }
        public double Score { get; set; }
    }
}
