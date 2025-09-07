using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp;
using OpenCvSharpEx.Internal;

namespace OpenCvSharpEx
{
    /// <summary>
    /// Specifies the method used for refining match results.
    /// </summary>
    public enum RefinementMethod
    {
        /// <summary>No refinement is performed. The result from the coarse search is returned. Fastest.</summary>
        None,
        /// <summary>Uses quadratic interpolation on a Normalized Cross-Correlation (NCC) score map in a small neighborhood. Fast and provides sub-pixel accuracy.</summary>
        Quadratic,
        /// <summary>Uses Iterative Closest Point (ICP) algorithm with a K-D tree for the most accurate alignment. Slowest.</summary>
        ICP,
        /// <summary>Uses Iterative Closest Point (ICP) algorithm with an edge distance map. Can be more robust than K-D tree for noisy images.</summary>
        ICPEdge,
        /// <summary>A fast variant of quadratic refinement that interpolates angle from neighboring template scores. Much faster than standard Quadratic.</summary>
        FastQuadratic
    }
    public class ShapeMatcher : IDisposable
    {
        public ShapeMatcher()
        {

        }
        private IntPtr shapeMatcherObj;
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
        public int Features
        {
            get;
            set;
        } = 63;
        public int PyramidLevels
        {
            get;
            set;
        } = 2;
        public bool UseFusion
        {
            get;
            set;
        }
        /// <summary>
        /// Gets or sets the method used to refine the position of found matches. Default is None.
        /// </summary>
        public RefinementMethod Refinement { get; set; } = RefinementMethod.None;

        public void Teach(Mat pattern)
        {
            var ret = NativeMethods.shapematcher_ShapeMatcher_new(pattern.CvPtr, this.MinAngle, this.MaxAngle, this.AngleStep, this.Features, this.PyramidLevels, out this.shapeMatcherObj);
        }
        public Feature[] GetFeatures(int templateIndex)
        {
            if (this.shapeMatcherObj == IntPtr.Zero)
                throw new OpenCvSharpException("No pattern is taught yet.");
            NativeMethods.shapematcher_ShapeMatcher_getFeatures(this.shapeMatcherObj, templateIndex, IntPtr.Zero, out var featuresCount);
            if (featuresCount > 0)
            {
                var features = new Feature[featuresCount];
                NativeMethods.shapematcher_ShapeMatcher_getFeatures(this.shapeMatcherObj, templateIndex, features, out _);
                return features;
            }
            else return new Feature[0];
        }
        public ShapeTemplate GetTemplate(int templateIndex)
        {
            if (this.shapeMatcherObj == IntPtr.Zero)
                throw new OpenCvSharpException("No pattern is taught yet.");

            NativeMethods.shapematcher_ShapeMatcher_getTemplate(this.shapeMatcherObj, templateIndex, out _, out _, IntPtr.Zero, out var featuresCount);

            Feature[] features;
            float angle, scale;

            if (featuresCount > 0)
            {
                features = new Feature[featuresCount];
                NativeMethods.shapematcher_ShapeMatcher_getTemplate(this.shapeMatcherObj, templateIndex, out angle, out scale, features, out _);
            }
            else
            {
                features = new Feature[0];
                NativeMethods.shapematcher_ShapeMatcher_getTemplate(this.shapeMatcherObj, templateIndex, out angle, out scale, null, out _);
            }

            return new ShapeTemplate
            {
                Angle = angle,
                Scale = scale,
                Features = features
            };
        }
        public Point GetPatternOffset()
        {
            if (this.shapeMatcherObj == IntPtr.Zero)
                throw new OpenCvSharpException("No pattern is taught yet.");
            NativeMethods.shapematcher_ShapeMatcher_getPatternOffset(this.shapeMatcherObj, out var offset);
            return offset;
        }
        public void PreprocessPattern()
        {

        }
        public ShapeMatcherResults Search(Mat image)
        {
            if (this.shapeMatcherObj == IntPtr.Zero)
                throw new OpenCvSharpException("No pattern is taught yet.");
            double score = this.AcceptancePercentage;
            var ret = this.UseFusion ? NativeMethods.shapematcher_ShapeMatcher_searchFusion(this.shapeMatcherObj, image.CvPtr, (int)this.Refinement, out var location, out var angle, ref score, out var templateID, out var rotatedBounds)
                : NativeMethods.shapematcher_ShapeMatcher_search(this.shapeMatcherObj, image.CvPtr, (int)this.Refinement, out location, out angle, ref score, out templateID, out rotatedBounds);
            var results = new ShapeMatcherResults()
            {
                Location = location,
                Angle = angle,
                Score = score,
                TemplateID = templateID,
                RotatedBounds = rotatedBounds,
            };
            results.Bounds = results.RotatedBounds.BoundingRect2d();
            return results;
        }
        public Mat GetPaddedPattern(double angle)
        {
            if (this.shapeMatcherObj == IntPtr.Zero)
                throw new OpenCvSharpException("No pattern is taught yet.");
            var paddedPattern = new Mat();
            NativeMethods.shapematcher_ShapeMatcher_getPaddedPattern(this.shapeMatcherObj, angle, paddedPattern.CvPtr);
            return paddedPattern;
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
        public int TemplateID { get; set; }
    }
    public class ShapeTemplate
    {
        public double Angle { get; set; }
        public double Scale { get; set; }
        public Feature[] Features { get; set; }
    }
    public struct Feature
    {
        public int x;
        public int y;
        public int label;
        public float theta;
    }
}
