﻿using System;
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
        public void Teach(Mat pattern)
        {
            var ret = NativeMethods.shapematcher_ShapeMatcher_new(pattern.CvPtr, this.MinAngle, this.MaxAngle, this.AngleStep, this.AcceptancePercentage, this.Features, this.PyramidLevels, out this.shapeMatcherObj);
        }
        public Feature[] GetFeatures(int templateIndex)
        {
            if (this.shapeMatcherObj == null)
                throw new OpenCvSharpException("No pattern is taught yet.");
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
        public ShapeMatcherResults Search(Mat image, bool refineResults = false)
        {
            if (this.shapeMatcherObj == IntPtr.Zero)
                throw new OpenCvSharpException("No pattern is taught yet.");
            double score = this.AcceptancePercentage;
            var ret = this.UseFusion ? NativeMethods.shapematcher_ShapeMatcher_searchFusion(this.shapeMatcherObj, image.CvPtr, refineResults, out var location, out double angle, ref score, out int templateID)
                : NativeMethods.shapematcher_ShapeMatcher_search(this.shapeMatcherObj, image.CvPtr, refineResults, out location, out angle, ref score, out templateID);
            return new ShapeMatcherResults()
            {
                Location = location,
                Angle = angle,
                Score = score,
                TemplateID = templateID
            };
        }
        public Mat GetPaddedPattern(double angle)
        {
            if (this.shapeMatcherObj == IntPtr.Zero)
                throw new OpenCvSharpException("No pattern is taught yet.");
            var ret = NativeMethods.shapematcher_ShapeMatcher_getPaddedPattern(this.shapeMatcherObj, angle, out var ptr);
            return new Mat(ptr);
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
    public struct Feature
    {
        public int x;
        public int y;
        public int label;
        public float theta;
    }
}
