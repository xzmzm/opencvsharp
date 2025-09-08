using System;
using System.Runtime.InteropServices;
using OpenCvSharp;
using OpenCvSharpEx.Internal;

namespace OpenCvSharpEx
{
    /// <summary>
    /// Represents a contour with sub-pixel accurate points.
    /// </summary>
    public class Contour
    {
        /// <summary>
        /// The points of the contour with sub-pixel accuracy.
        /// </summary>
        public Point2f[] Points { get; internal set; }

        /// <summary>
        /// The direction of the edge at each point (in radians).
        /// </summary>
        public float[] Direction { get; internal set; }

        /// <summary>
        /// The response (magnitude) of the edge at each point.
        /// </summary>
        public float[] Response { get; internal set; }
    }

    // Internal struct for marshaling from native code
    [StructLayout(LayoutKind.Sequential)]
    internal struct ContourC
    {
        public IntPtr Points;
        public int NumPoints;
        public IntPtr Direction;
        public IntPtr Response;
    }

    public static partial class Cv2Ex
    {
        /// <summary>
        /// Finds edges in an image using a sub-pixel accurate algorithm.
        /// </summary>
        /// <param name="gray">Input 8-bit single-channel image.</param>
        /// <param name="alpha">The alpha parameter for the Gaussian filter (sigma).</param>
        /// <param name="low">The lower hysteresis threshold.</param>
        /// <param name="high">The higher hysteresis threshold.</param>
        /// <param name="contours">Detected contours. Each contour is a vector of points.</param>
        /// <param name="hierarchy">Optional output vector containing information about the image topology.</param>
        /// <param name="mode">Contour retrieval mode.</param>
        public static void EdgesSubPix(
            InputArray gray,
            double alpha,
            int low,
            int high,
            out Contour[] contours,
            OutputArray hierarchy,
            RetrievalModes mode)
        {
            if (gray == null)
                throw new ArgumentNullException(nameof(gray));
            gray.ThrowIfDisposed();

            var grayMat = gray.GetMat();
            if (grayMat.Channels() != 1 || grayMat.Depth() != MatType.CV_8U)
                throw new ArgumentException("Input image must be 8-bit single-channel.", nameof(gray));

            IntPtr hierarchyPtr = hierarchy?.CvPtr ?? IntPtr.Zero;

            var ret = NativeMethods.cv2ex_EdgesSubPix(
                grayMat.CvPtr, alpha, low, high,
                out var contoursPtr, out var numContours,
                hierarchyPtr, (int)mode);

            //Cv2.CheckStatus(ret);

            if (numContours > 0 && contoursPtr != IntPtr.Zero)
            {
                contours = new Contour[numContours];
                var contourCSize = Marshal.SizeOf<ContourC>();

                for (int i = 0; i < numContours; i++)
                {
                    IntPtr currentContourPtr = new IntPtr(contoursPtr.ToInt64() + i * contourCSize);
                    var contourC = Marshal.PtrToStructure<ContourC>(currentContourPtr);

                    var contour = new Contour
                    {
                        Points = new Point2f[contourC.NumPoints],
                        Direction = new float[contourC.NumPoints],
                        Response = new float[contourC.NumPoints]
                    };

                    if (contourC.NumPoints > 0)
                    {
                        // Marshal points
                        var point2fSize = Marshal.SizeOf<Point2f>();
                        for (int j = 0; j < contourC.NumPoints; j++)
                        {
                            IntPtr p = new IntPtr(contourC.Points.ToInt64() + j * point2fSize);
                            contour.Points[j] = Marshal.PtrToStructure<Point2f>(p);
                        }

                        // Marshal direction and response
                        Marshal.Copy(contourC.Direction, contour.Direction, 0, contourC.NumPoints);
                        Marshal.Copy(contourC.Response, contour.Response, 0, contourC.NumPoints);
                    }

                    contours[i] = contour;
                }

                // Free the memory allocated in C++
                NativeMethods.cv2ex_FreeContours(contoursPtr, numContours);
            }
            else
            {
                contours = Array.Empty<Contour>();
            }
        }

        /// <summary>
        /// Finds edges in an image using a sub-pixel accurate algorithm.
        /// </summary>
        /// <param name="gray">Input 8-bit single-channel image.</param>
        /// <param name="alpha">The alpha parameter for the Gaussian filter (sigma).</param>
        /// <param name="low">The lower hysteresis threshold.</param>
        /// <param name="high">The higher hysteresis threshold.</param>
        /// <param name="contours">Detected contours. Each contour is a vector of points.</param>
        public static void EdgesSubPix(
            InputArray gray,
            double alpha,
            int low,
            int high,
            out Contour[] contours)
        {
            EdgesSubPix(gray, alpha, low, high, out contours, null, RetrievalModes.List);
        }
    }
}
