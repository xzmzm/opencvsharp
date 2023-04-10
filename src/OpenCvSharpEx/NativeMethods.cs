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
        public static extern ExceptionStatus shapematcher_ShapeMatcher_new(IntPtr pattern, double minAngle, double maxAngle, double angleStep, double acceptancePercentage, out IntPtr shapeMatcher);

        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_delete(IntPtr shapeMatcherObj);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_teach(IntPtr shapeMatcherObj, IntPtr pattern);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_search(IntPtr shapeMatcherObj, IntPtr image, out OpenCvSharp.Point2d location, out double angle);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_getPaddedPattern(IntPtr shapeMatcherObj, double angle, out IntPtr paddedPattern);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_getFeaturesCount(IntPtr shapeMatcherObj, int templateIndex, out int featuresCount);
        [Pure, DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus shapematcher_ShapeMatcher_getFeatures(IntPtr shapeMatcherObj, int templateIndex, [Out] Feature[] features);
  }
}
