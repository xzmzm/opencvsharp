using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp;
using OpenCvSharpEx.Internal;

namespace OpenCvSharpEx
{
    public class ShapeMatcher : IDisposable
    {
        public ShapeMatcher()
        {

        }
        IntPtr shapeMatcherObj;
        public double AcceptancePercentage
        {
            get;
            set;
        } = 70.0;
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
        public void Teach(Mat pattern)
        {
            var ret = NativeMethods.shapematcher_ShapeMatcher_new(pattern.CvPtr, this.MinAngle, this.MaxAngle, this.AngleStep, this.AcceptancePercentage, out this.shapeMatcherObj);
        }
        public Feature[] GetFeatures(int templateIndex)
        {
            NativeMethods.shapematcher_ShapeMatcher_getFeaturesCount(this.shapeMatcherObj, templateIndex, out var featuresCount);
            if (featuresCount > 0)
            {
                var features = new Feature[featuresCount];
                NativeMethods.shapematcher_ShapeMatcher_getFeatures(this.shapeMatcherObj, templateIndex, features);
                return features;
            }
            else return new Feature[0];
        }
        public void PreprocessPattern()
        {

        }
        public ShapeMatcherResults Search(Mat image)
        {
            var ret = NativeMethods.shapematcher_ShapeMatcher_search(this.shapeMatcherObj, image.CvPtr, out var location, out double angle);
            return new ShapeMatcherResults()
            {
                Location = location,
                Angle = angle
            };
        }
        ~ShapeMatcher()
        {
            this.Dispose();
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            if (this.shapeMatcherObj != IntPtr.Zero)
            {
                NativeMethods.shapematcher_ShapeMatcher_delete(this.shapeMatcherObj);
                this.shapeMatcherObj = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }
    }
    public class ShapeMatcherResults
    {
        public Point2d Location { get; set; }
        public double Angle { get; set; }
        public RotatedRect RotatedBounds { get; set; }
        public Rect2d Bounds { get; set; }
        public double Score { get; set; }
    }
    public struct Feature
    {
        public int x;
        public int y;
        public int label;
        public float theta;
    }
}
