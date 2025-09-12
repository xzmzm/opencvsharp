using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp;
using OpenCvSharpEx.Internal;

namespace OpenCvSharpEx
{
    public class RotatedPatternMatcher : IDisposable
    {
        private IntPtr rotatedPatternMatcherObj;
        public RotatedPatternMatcher()
        {
        }

        public void Teach(Mat pattern, int pyramidLevels)
        {
            if (this.rotatedPatternMatcherObj != IntPtr.Zero)
            {
                NativeMethods.rotatedPatternMatcher_RotatedPatternMatcher_delete(this.rotatedPatternMatcherObj);
                this.rotatedPatternMatcherObj = IntPtr.Zero;
            }

            NativeMethods.rotatedPatternMatcher_RotatedPatternMatcher_new(out this.rotatedPatternMatcherObj);
            if (this.rotatedPatternMatcherObj == IntPtr.Zero)
                throw new OpenCvSharpException("Failed to create native RotatedPatternMatcher object.");

            NativeMethods.rotatedPatternMatcher_RotatedPatternMatcher_teach(this.rotatedPatternMatcherObj, pattern.CvPtr, pyramidLevels);
        }

        public RotationPatternMatcherResults[] Search(
            Mat image,
            double acceptanceScore,
            double minAngle,
            double maxAngle,
            double angleStep,
            int maxMatchCount,
            double maxOverlapRatio,
            int matchCandidateCount)
        {
            if (this.rotatedPatternMatcherObj == IntPtr.Zero)
                throw new OpenCvSharpException("No pattern is taught yet.");

            var ret = NativeMethods.rotatedPatternMatcher_RotatedPatternMatcher_search(
                this.rotatedPatternMatcherObj, image.CvPtr, acceptanceScore, minAngle, maxAngle, angleStep, maxMatchCount, maxOverlapRatio, matchCandidateCount,
                out IntPtr results, out int resultsLength);

            var r = new RotationPatternMatcherResults[resultsLength];
            var p = results;
            for (int i = 0; i < r.Length; ++i)
            {
                r[i] = (RotationPatternMatcherResults)System.Runtime.InteropServices.Marshal.PtrToStructure(p, typeof(RotationPatternMatcherResults));
                r[i].Bounds = r[i].RotatedBounds.BoundingRect2d();
                p = new IntPtr(p.ToInt64() + System.Runtime.InteropServices.Marshal.SizeOf(typeof(RotationPatternMatcherResults)));
            }
            System.Runtime.InteropServices.Marshal.FreeCoTaskMem(results);
            return r;
        }

        ~RotatedPatternMatcher()
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
