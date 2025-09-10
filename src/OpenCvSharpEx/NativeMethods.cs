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
        public static extern ExceptionStatus shapematcher_ShapeMatcher_new(IntPtr pattern, double minAngle, double maxAngle, double angleStep, int nFeatures, int pyramidLevels, out IntPtr shapeMatcher);

        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_delete(IntPtr shapeMatcherObj);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_teach(IntPtr shapeMatcherObj, IntPtr pattern, int nFeatures, int pyramidLevels);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_search(IntPtr shapeMatcherObj, IntPtr image, int refinementLevel, out OpenCvSharp.Point2d location, out double angle, ref double score, out int templateID, out OpenCvSharp.RotatedRect rotatedBounds);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_searchFusion(IntPtr shapeMatcherObj, IntPtr image, int refinementLevel, out OpenCvSharp.Point2d location, out double angle, ref double score, out int templateID, out OpenCvSharp.RotatedRect rotatedBounds);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_getPaddedPattern(IntPtr shapeMatcherObj, double angle, IntPtr outPaddedPattern);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_getTemplate(IntPtr shapeMatcherObj, int templateIndex, out float angle, out float scale, IntPtr features, out int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_getTemplate(IntPtr shapeMatcherObj, int templateIndex, out float angle, out float scale, [Out] Feature[] features, out int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_getPatternOffset(IntPtr shapeMatcherObj, out OpenCvSharp.Point offset);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_getFeatures(IntPtr shapeMatcherObj, int templateIndex, IntPtr features, out int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_getFeatures(IntPtr shapeMatcherObj, int templateIndex, [Out] Feature[] features, out int count);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus rotatedPatternMatcher_RotatedPatternMatcher_new(out IntPtr rotatedPatternMatcher);

        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus rotatedPatternMatcher_RotatedPatternMatcher_delete(IntPtr shapeMatcherObj);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus rotatedPatternMatcher_RotatedPatternMatcher_teach(IntPtr shapeMatcherObj, IntPtr pattern, int pyramidLevels);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus rotatedPatternMatcher_RotatedPatternMatcher_search(IntPtr shapeMatcherObj, IntPtr image, double acceptancePercentage, double minAngle, double maxAngle, double angleStep, int maxMatchCount, double maxOverlapRatio,
            out IntPtr rotationPatternMatcherResults, out int rotationPatternMatcherResultsLength);
    }

    public static partial class NativeMethods
    {
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cv2ex_EdgesSubPix(IntPtr gray, double alpha, int low, int high, out IntPtr contours, out int numContours, IntPtr hierarchy, int mode);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cv2ex_FreeContours(IntPtr contours, int numContours);

        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cv2ex_PrecomputeEdgesSubPix(IntPtr gray, double alpha, IntPtr dx, IntPtr dy);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cv2ex_PrecomputeEdgesSubPixBilateral(IntPtr gray, int d, double sigmaColor, double sigmaSpace,
            double gradientAlpha, IntPtr dx, IntPtr dy);

        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern ExceptionStatus cv2ex_RefineContourSubPix(
            IntPtr dx, IntPtr dy, [In] OpenCvSharp.Point[] initialContour, int contourLength, int searchRadius, [MarshalAs(UnmanagedType.I1)] bool fixCorners,
            out ContourC outRefinedContour);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern ExceptionStatus cv2ex_RefineContoursSubPix(
            IntPtr dx, IntPtr dy, [In] OpenCvSharp.Point[] initialContoursData, [In] int[] contourLengths, int numContours,
            int searchRadius, [MarshalAs(UnmanagedType.I1)] bool fixCorners,
            out IntPtr outRefinedContours, out int outNumContours);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern ExceptionStatus cv2ex_FreeContourData(IntPtr contourDataPtr);

        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern ExceptionStatus cv2ex_PrecomputeGradientsSobel(
            IntPtr gray, IntPtr gradX, IntPtr gradY, int ksize);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern ExceptionStatus cv2ex_RefineContourCentroid(
            IntPtr gradX, IntPtr gradY, [In] OpenCvSharp.Point[] initialContour, int contourLength, int windowSize,
            out ContourC outRefinedContour);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        internal static extern ExceptionStatus cv2ex_RefineContoursCentroid(
            IntPtr gradX, IntPtr gradY, [In] OpenCvSharp.Point[] initialContoursData, [In] int[] contourLengths, int numContours,
            int windowSize,
            out IntPtr outRefinedContours, out int outNumContours);
    }
}
