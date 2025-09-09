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
        /// <param name="hierarchy">Optional output vector containing information about the image topology.</param>
        /// <param name="mode">Contour retrieval mode.</param>
        /// <returns>Detected contours. Each contour is a vector of points.</returns>
        public static Contour[] EdgesSubPix(
            InputArray gray,
            double alpha,
            int low,
            int high,
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
                var contours = new Contour[numContours];
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
                return contours;
            }
            else
            {
                return Array.Empty<Contour>();
            }
        }

        /// <summary>
        /// Finds edges in an image using a sub-pixel accurate algorithm.
        /// </summary>
        /// <param name="gray">Input 8-bit single-channel image.</param>
        /// <param name="alpha">The alpha parameter for the Gaussian filter (sigma).</param>
        /// <param name="low">The lower hysteresis threshold.</param>
        /// <param name="high">The higher hysteresis threshold.</param>
        /// <returns>Detected contours. Each contour is a vector of points.</returns>
        public static Contour[] EdgesSubPix(
            InputArray gray,
            double alpha,
            int low,
            int high)
        {
            return EdgesSubPix(gray, alpha, low, high, null, RetrievalModes.List);
        }

        /// <summary>
        /// Precomputes the gradient maps (dx, dy) required for sub-pixel edge refinement.
        /// This is useful when refining multiple contours on the same image to avoid redundant computations.
        /// </summary>
        /// <param name="gray">Input 8-bit single-channel image.</param>
        /// <param name="alpha">The alpha parameter for the Gaussian filter (sigma).</param>
        /// <param name="dx">Output 16-bit signed integer (CV_16S) gradient map in X direction.</param>
        /// <param name="dy">Output 16-bit signed integer (CV_16S) gradient map in Y direction.</param>
        public static void PrecomputeEdgesSubPix(InputArray gray, double alpha, OutputArray dx, OutputArray dy)
        {
            if (gray == null) throw new ArgumentNullException(nameof(gray));
            if (dx == null) throw new ArgumentNullException(nameof(dx));
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            gray.ThrowIfDisposed();
            dx.ThrowIfNotReady();
            dy.ThrowIfNotReady();

            Mat grayMat = gray.GetMat();
            Mat dxMat = dx.GetMat();
            Mat dyMat = dy.GetMat();
            NativeMethods.cv2ex_PrecomputeEdgesSubPix(grayMat.CvPtr, alpha, dxMat.CvPtr, dyMat.CvPtr);

            GC.KeepAlive(gray);
            GC.KeepAlive(dx);
            GC.KeepAlive(dy);
            GC.KeepAlive(grayMat);
            GC.KeepAlive(dxMat);
            GC.KeepAlive(dyMat);
        }

        /// <summary>
        /// Precomputes the gradient maps (dx, dy) using an edge-preserving bilateral filter.
        /// This is useful when refining multiple contours on the same image to avoid redundant computations
        /// and to prevent the inward shift of edges on curved objects caused by Gaussian blurring.
        /// </summary>
        /// <param name="gray">Input 8-bit single-channel image.</param>
        /// <param name="diameter">Diameter of each pixel neighborhood that is used during filtering.</param>
        /// <param name="sigmaColor">Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.</param>
        /// <param name="sigmaSpace">Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough. </param>
        /// <param name="gradientAlpha">The sigma parameter for the Canny-style gradient kernel applied after smoothing.</param>
        /// <param name="dx">Output 16-bit signed integer (CV_16S) gradient map in X direction.</param>
        /// <param name="dy">Output 16-bit signed integer (CV_16S) gradient map in Y direction.</param>
        public static void PrecomputeEdgesSubPixBilateral(
            InputArray gray,
            int diameter,
            double sigmaColor,
            double sigmaSpace,
            double gradientAlpha,
            OutputArray dx,
            OutputArray dy)
        {
            if (gray == null) throw new ArgumentNullException(nameof(gray));
            if (dx == null) throw new ArgumentNullException(nameof(dx));
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            gray.ThrowIfDisposed();
            dx.ThrowIfNotReady();
            dy.ThrowIfNotReady();

            Mat grayMat = gray.GetMat();
            Mat dxMat = dx.GetMat();
            Mat dyMat = dy.GetMat();
            NativeMethods.cv2ex_PrecomputeEdgesSubPixBilateral(grayMat.CvPtr, diameter, sigmaColor, sigmaSpace, gradientAlpha, dxMat.CvPtr, dyMat.CvPtr);

            GC.KeepAlive(gray);
            GC.KeepAlive(dx);
            GC.KeepAlive(dy);
            GC.KeepAlive(grayMat);
            GC.KeepAlive(dxMat);
            GC.KeepAlive(dyMat);
        }

        /// <summary>
        /// Refines a given integer-precision contour to sub-pixel accuracy.
        /// </summary>
        /// <param name="dx">Precomputed 16-bit signed integer (CV_16S) gradient map in X direction.</param>
        /// <param name="dy">Precomputed 16-bit signed integer (CV_16S) gradient map in Y direction.</param>
        /// <param name="initialContour">The integer-precision contour to refine.</param>
        /// <param name="searchRadius">The radius (in pixels) to search for the strongest edge along the normal of each contour point.</param>
        /// <param name="fixCorners">Whether to apply a smoothing filter at sharp corners to prevent self-intersections. Default is false.</param>
        /// <returns>The output sub-pixel accurate contour.</returns>
        public static Contour RefineContourSubPix(
            InputArray dx,
            InputArray dy,
            Point[] initialContour,
            int searchRadius,
            bool fixCorners = false)
        {
            if (dx == null) throw new ArgumentNullException(nameof(dx));
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            if (initialContour == null) throw new ArgumentNullException(nameof(initialContour));
            dx.ThrowIfDisposed();
            dy.ThrowIfDisposed();

            Mat dxMat = dx.GetMat();
            Mat dyMat = dy.GetMat();

            var contourC = new ContourC();
            Contour refinedContour;
            try
            {
                NativeMethods.cv2ex_RefineContourSubPix(
                    dxMat.CvPtr, dyMat.CvPtr, initialContour, initialContour.Length, searchRadius, fixCorners,
                    ref contourC);

                refinedContour = new Contour();
                if (contourC.NumPoints > 0)
                {
                    refinedContour.Points = new Point2f[contourC.NumPoints];
                    refinedContour.Direction = new float[contourC.NumPoints];
                    refinedContour.Response = new float[contourC.NumPoints];

                    var point2fSize = Marshal.SizeOf<Point2f>();
                    for (int j = 0; j < contourC.NumPoints; j++)
                    {
                        IntPtr p = new IntPtr(contourC.Points.ToInt64() + j * point2fSize);
                        refinedContour.Points[j] = Marshal.PtrToStructure<Point2f>(p);
                    }

                    Marshal.Copy(contourC.Direction, refinedContour.Direction, 0, contourC.NumPoints);
                    Marshal.Copy(contourC.Response, refinedContour.Response, 0, contourC.NumPoints);
                }
                else
                {
                    refinedContour.Points = Array.Empty<Point2f>();
                    refinedContour.Direction = Array.Empty<float>();
                    refinedContour.Response = Array.Empty<float>();
                }
            }
            finally
            {
                NativeMethods.cv2ex_FreeContourData(ref contourC);
            }
            GC.KeepAlive(dx);
            GC.KeepAlive(dy);
            GC.KeepAlive(dxMat);
            GC.KeepAlive(dyMat);
            return refinedContour;
        }

        /// <summary>
        /// Refines a given integer-precision contour to sub-pixel accuracy.
        /// This is a convenience overload that computes gradient maps internally.
        /// </summary>
        /// <param name="gray">Input 8-bit single-channel image.</param>
        /// <param name="initialContour">The integer-precision contour to refine.</param>
        /// <param name="alpha">The alpha parameter for the Gaussian filter (sigma).</param>
        /// <param name="searchRadius">The radius (in pixels) to search for the strongest edge along the normal of each contour point.</param>
        /// <param name="fixCorners">Whether to apply a smoothing filter at sharp corners to prevent self-intersections. Default is false.</param>
        /// <returns>The output sub-pixel accurate contour.</returns>
        public static Contour RefineContourSubPix(
            InputArray gray,
            Point[] initialContour,
            double alpha,
            int searchRadius,
            bool fixCorners = false)
        {
            using (var dx = new Mat())
            using (var dy = new Mat())
            {
                PrecomputeEdgesSubPix(gray, alpha, dx, dy);
                return RefineContourSubPix(dx, dy, initialContour, searchRadius, fixCorners);
            }
        }

        /// <summary>
        /// Refines a given integer-precision contour to sub-pixel accuracy using an edge-preserving bilateral filter for preprocessing.
        /// </summary>
        /// <param name="gray">Input 8-bit single-channel image.</param>
        /// <param name="initialContour">The integer-precision contour to refine.</param>
        /// <param name="diameter">Diameter of each pixel neighborhood that is used during filtering.</param>
        /// <param name="sigmaColor">Filter sigma in the color space.</param>
        /// <param name="sigmaSpace">Filter sigma in the coordinate space.</param>
        /// <param name="gradientAlpha">The sigma parameter for the Canny-style gradient kernel applied after smoothing.</param>
        /// <param name="searchRadius">The radius (in pixels) to search for the strongest edge along the normal of each contour point.</param>
        /// <param name="fixCorners">Whether to apply a smoothing filter at sharp corners to prevent self-intersections. Default is false.</param>
        /// <returns>The output sub-pixel accurate contour.</returns>
        public static Contour RefineContourSubPixBilateral(
            InputArray gray,
            Point[] initialContour,
            int diameter,
            double sigmaColor,
            double sigmaSpace,
            double gradientAlpha,
            int searchRadius,
            bool fixCorners = false)
        {
            using (var dx = new Mat())
            using (var dy = new Mat())
            {
                PrecomputeEdgesSubPixBilateral(gray, diameter, sigmaColor, sigmaSpace, gradientAlpha, dx, dy);
                return RefineContourSubPix(dx, dy, initialContour, searchRadius, fixCorners);
            }
        }

        /// <summary>
        /// Refines given integer-precision contours to sub-pixel accuracy.
        /// </summary>
        /// <param name="dx">Precomputed 16-bit signed integer (CV_16S) gradient map in X direction.</param>
        /// <param name="dy">Precomputed 16-bit signed integer (CV_16S) gradient map in Y direction.</param>
        /// <param name="initialContours">The integer-precision contours to refine.</param>
        /// <param name="searchRadius">The radius (in pixels) to search for the strongest edge along the normal of each contour point.</param>
        /// <param name="fixCorners">Whether to apply a smoothing filter at sharp corners to prevent self-intersections. Default is false.</param>
        /// <returns>The output sub-pixel accurate contours.</returns>
        public static Contour[] RefineContourSubPix(
            InputArray dx,
            InputArray dy,
            System.Collections.Generic.IEnumerable<Point[]> initialContours,
            int searchRadius,
            bool fixCorners = false)
        {
            if (dx == null) throw new ArgumentNullException(nameof(dx));
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            if (initialContours == null) throw new ArgumentNullException(nameof(initialContours));
            dx.ThrowIfDisposed();
            dy.ThrowIfDisposed();

            var initialContoursArray = System.Linq.Enumerable.ToArray(initialContours);
            int numContours = initialContoursArray.Length;
            if (numContours == 0)
            {
                return Array.Empty<Contour>();
            }

            var contourLengths = new int[numContours];
            int totalPoints = 0;
            for (int i = 0; i < numContours; i++)
            {
                contourLengths[i] = initialContoursArray[i]?.Length ?? 0;
                totalPoints += contourLengths[i];
            }

            var contoursData = new Point[totalPoints];
            int currentPos = 0;
            for (int i = 0; i < numContours; i++)
            {
                if (contourLengths[i] > 0)
                {
                    Array.Copy(initialContoursArray[i], 0, contoursData, currentPos, contourLengths[i]);
                    currentPos += contourLengths[i];
                }
            }

            Mat dxMat = dx.GetMat();
            Mat dyMat = dy.GetMat();

            NativeMethods.cv2ex_RefineContoursSubPix(
                dxMat.CvPtr, dyMat.CvPtr,
                contoursData, contourLengths, numContours,
                searchRadius, fixCorners,
                out var contoursPtr, out var outNumContours);

            Contour[] refinedContours;
            if (outNumContours > 0 && contoursPtr != IntPtr.Zero)
            {
                refinedContours = new Contour[outNumContours];
                var contourCSize = Marshal.SizeOf<ContourC>();

                for (int i = 0; i < outNumContours; i++)
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
                        var point2fSize = Marshal.SizeOf<Point2f>();
                        for (int j = 0; j < contourC.NumPoints; j++)
                        {
                            IntPtr p = new IntPtr(contourC.Points.ToInt64() + j * point2fSize);
                            contour.Points[j] = Marshal.PtrToStructure<Point2f>(p);
                        }

                        Marshal.Copy(contourC.Direction, contour.Direction, 0, contourC.NumPoints);
                        Marshal.Copy(contourC.Response, contour.Response, 0, contourC.NumPoints);
                    }
                    refinedContours[i] = contour;
                }
                NativeMethods.cv2ex_FreeContours(contoursPtr, outNumContours);
            }
            else
            {
                refinedContours = Array.Empty<Contour>();
            }

            GC.KeepAlive(dx);
            GC.KeepAlive(dy);
            GC.KeepAlive(dxMat);
            GC.KeepAlive(dyMat);
            return refinedContours;
        }

        /// <summary>
        /// Refines given integer-precision contours to sub-pixel accuracy.
        /// This is a convenience overload that computes gradient maps internally.
        /// </summary>
        /// <param name="gray">Input 8-bit single-channel image.</param>
        /// <param name="initialContours">The integer-precision contours to refine.</param>
        /// <param name="alpha">The alpha parameter for the Gaussian filter (sigma).</param>
        /// <param name="searchRadius">The radius (in pixels) to search for the strongest edge along the normal of each contour point.</param>
        /// <param name="fixCorners">Whether to apply a smoothing filter at sharp corners to prevent self-intersections. Default is false.</param>
        /// <returns>The output sub-pixel accurate contours.</returns>
        public static Contour[] RefineContourSubPix(
            InputArray gray,
            System.Collections.Generic.IEnumerable<Point[]> initialContours,
            double alpha,
            int searchRadius,
            bool fixCorners = false)
        {
            using (var dx = new Mat())
            using (var dy = new Mat())
            {
                PrecomputeEdgesSubPix(gray, alpha, dx, dy);
                return RefineContourSubPix(dx, dy, initialContours, searchRadius, fixCorners);
            }
        }

        /// <summary>
        /// Refines given integer-precision contours to sub-pixel accuracy using an edge-preserving bilateral filter for preprocessing.
        /// </summary>
        /// <param name="gray">Input 8-bit single-channel image.</param>
        /// <param name="initialContours">The integer-precision contours to refine.</param>
        /// <param name="diameter">Diameter of each pixel neighborhood that is used during filtering.</param>
        /// <param name="sigmaColor">Filter sigma in the color space.</param>
        /// <param name="sigmaSpace">Filter sigma in the coordinate space.</param>
        /// <param name="gradientAlpha">The sigma parameter for the Canny-style gradient kernel applied after smoothing.</param>
        /// <param name="searchRadius">The radius (in pixels) to search for the strongest edge along the normal of each contour point.</param>
        /// <param name="fixCorners">Whether to apply a smoothing filter at sharp corners to prevent self-intersections. Default is false.</param>
        /// <returns>The output sub-pixel accurate contours.</returns>
        public static Contour[] RefineContourSubPixBilateral(
            InputArray gray,
            System.Collections.Generic.IEnumerable<Point[]> initialContours,
            int diameter,
            double sigmaColor,
            double sigmaSpace,
            double gradientAlpha,
            int searchRadius,
            bool fixCorners = false)
        {
            using(var dx = new Mat())
            using (var dy = new Mat())
            {
                PrecomputeEdgesSubPixBilateral(gray, diameter, sigmaColor, sigmaSpace, gradientAlpha, dx, dy);
                return RefineContourSubPix(dx, dy, initialContours, searchRadius, fixCorners);
            }
        }
    }
}
