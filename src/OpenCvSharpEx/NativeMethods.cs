using System;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;

namespace OpenCvSharpEx.Internal
{
    /// <summary>
    /// Whether native methods for P/Invoke raises an exception
    /// </summary>
    public enum ExceptionStatus
    {
#pragma warning disable 1591
        NotOccurred = 0,
        Occurred = 1
    }
    public static partial class NativeMethods
    {
        public const string DllExtern = "OpenCvSharpExtern";
        private const UnmanagedType StringUnmanagedTypeWindows = UnmanagedType.LPStr;

        private const UnmanagedType StringUnmanagedTypeNotWindows = UnmanagedType.LPStr;
//#if NET48 || NETSTANDARD2_0
//            UnmanagedType.LPStr;
//#else
//        UnmanagedType.LPUTF8Str;
//#endif
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_new(IntPtr pattern, double minAngle, double maxAngle, double angleStep, double acceptancePercentage, int nFeatures, int pyramidLevels, out IntPtr shapeMatcher);

        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_delete(IntPtr shapeMatcherObj);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_teach(IntPtr shapeMatcherObj, IntPtr pattern);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_search(IntPtr shapeMatcherObj, IntPtr image, bool refineResults, out OpenCvSharp.Point2d location, out double angle, ref double score, out int templateID);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_searchFusion(IntPtr shapeMatcherObj, IntPtr image, bool refineResults, out OpenCvSharp.Point2d location, out double angle, ref double score, out int templateID);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_getPaddedPattern(IntPtr shapeMatcherObj, double angle, out IntPtr paddedPattern);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_getFeaturesCount(IntPtr shapeMatcherObj, int templateIndex, out int featuresCount);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_getFeatures(IntPtr shapeMatcherObj, int templateIndex, [Out] Feature[] features);


        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus rotatedPatternMatcher_RotatedPatternMatcher_new(IntPtr pattern, double minAngle, double maxAngle, double angleStep, int minReducedArea, out IntPtr rotatedPatternMatcher);

        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus rotatedPatternMatcher_RotatedPatternMatcher_delete(IntPtr shapeMatcherObj);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus rotatedPatternMatcher_RotatedPatternMatcher_teach(IntPtr shapeMatcherObj, IntPtr pattern, double minAngle, double maxAngle, double angleStep, int minReducedArea);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus rotatedPatternMatcher_RotatedPatternMatcher_search(IntPtr shapeMatcherObj, IntPtr image, double acceptancePercentage, double minAngle, double maxAngle, double angleStep, int maxMatchCount, int minReducedArea, double maxOverlapRatio,
            out IntPtr rotationPatternMatcherResults, out int rotationPatternMatcherResultsLength);
    }
}
