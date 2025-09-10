using System;
using System.Runtime.InteropServices;
using System.Security;
using OpenCvSharp;
using OpenCvSharpEx.Internal;

namespace OpenCvSharpEx
{
    /// <summary>
    /// Represents a contour with sub-pixel accurate points, wrapping unmanaged memory.
    /// This class must be disposed to release the underlying memory.
    /// </summary>
    public class Contour : IDisposable
    {
        private IntPtr dataPtr;
        private readonly int numPoints;
        private bool disposedValue;

        internal unsafe Contour(ContourC c)
        {
            this.numPoints = c.NumPoints;
            if (this.numPoints > 0)
            {
                // In C++, we allocated a single block and 'Points' points to the beginning of it.
                this.dataPtr = c.Points;
            }
            else
            {
                this.dataPtr = IntPtr.Zero;
            }
        }

        /// <summary>
        /// The number of points in the contour.
        /// </summary>
        public int Length => this.numPoints;

        /// <summary>
        /// Gets a value indicating whether the contour is empty.
        /// </summary>
        public bool IsEmpty => this.numPoints == 0;

        private enum ContourDataType
        {
            Points,
            NormalAngles,
            Response,
            IntPoints
        }

        private unsafe IntPtr GetDataPointer(ContourDataType type)
        {
            if (this.dataPtr == IntPtr.Zero)
                return IntPtr.Zero;

            var pointsBytes = (long)this.numPoints * sizeof(Point2f);
            var anglesBytes = (long)this.numPoints * sizeof(float);
            var responseBytes = (long)this.numPoints * sizeof(float);

            switch (type)
            {
                case ContourDataType.Points: return this.dataPtr;
                case ContourDataType.NormalAngles: return this.dataPtr + (int)pointsBytes;
                case ContourDataType.Response: return this.dataPtr + (int)pointsBytes + (int)anglesBytes;
                case ContourDataType.IntPoints: return this.dataPtr + (int)pointsBytes + (int)anglesBytes + (int)responseBytes;
                default: throw new ArgumentOutOfRangeException(nameof(type));
            }
        }

        /// <summary>
        /// Gets a read-only span over the contour points.
        /// </summary>
        public unsafe ReadOnlySpan<Point2f> GetPoints()
        {
            if (this.disposedValue) throw new ObjectDisposedException(nameof(Contour));
            if (this.IsEmpty) return ReadOnlySpan<Point2f>.Empty;
            return new ReadOnlySpan<Point2f>(this.GetDataPointer(ContourDataType.Points).ToPointer(), this.numPoints);
        }

        /// <summary>
        /// Gets a read-only span over the normal angles.
        /// </summary>
        public unsafe ReadOnlySpan<float> GetNormalAngles()
        {
            if (this.disposedValue) throw new ObjectDisposedException(nameof(Contour));
            if (this.IsEmpty) return ReadOnlySpan<float>.Empty;
            return new ReadOnlySpan<float>(this.GetDataPointer(ContourDataType.NormalAngles).ToPointer(), this.numPoints);
        }

        /// <summary>
        /// Gets a read-only span over the edge responses.
        /// </summary>
        public unsafe ReadOnlySpan<float> GetResponse()
        {
            if (this.disposedValue) throw new ObjectDisposedException(nameof(Contour));
            if (this.IsEmpty) return ReadOnlySpan<float>.Empty;
            return new ReadOnlySpan<float>(this.GetDataPointer(ContourDataType.Response).ToPointer(), this.numPoints);
        }

        /// <summary>
        /// Gets a read-only span over the integer-precision points.
        /// </summary>
        public unsafe ReadOnlySpan<Point> GetIntPoints()
        {
            if (this.disposedValue) throw new ObjectDisposedException(nameof(Contour));
            if (this.IsEmpty) return ReadOnlySpan<Point>.Empty;
            return new ReadOnlySpan<Point>(this.GetDataPointer(ContourDataType.IntPoints).ToPointer(), this.numPoints);
        }

        /// <summary>
        /// Gets a copy of the contour points as an array.
        /// This method is provided for compatibility with consumers that do not support ReadOnlySpan&lt;T&gt; (e.g., IronPython).
        /// </summary>
        /// <returns>A new array containing the points.</returns>
        public Point2f[] GetPointsArray()
        {
            return this.GetPoints().ToArray();
        }

        /// <summary>
        /// Gets a copy of the normal angles as an array.
        /// This method is provided for compatibility with consumers that do not support ReadOnlySpan&lt;T&gt; (e.g., IronPython).
        /// </summary>
        /// <returns>A new array containing the normal angles.</returns>
        public float[] GetNormalAnglesArray()
        {
            return this.GetNormalAngles().ToArray();
        }

        /// <summary>
        /// Gets a copy of the edge responses as an array.
        /// This method is provided for compatibility with consumers that do not support ReadOnlySpan&lt;T&gt; (e.g., IronPython).
        /// </summary>
        /// <returns>A new array containing the edge responses.</returns>
        public float[] GetResponseArray()
        {
            return this.GetResponse().ToArray();
        }

        /// <summary>
        /// Gets a copy of the integer-precision points as an array.
        /// This method is provided for compatibility with consumers that do not support ReadOnlySpan&lt;T&gt; (e.g., IronPython).
        /// </summary>
        /// <returns>A new array containing the integer-precision points.</returns>
        public Point[] GetIntPointsArray()
        {
            return this.GetIntPoints().ToArray();
        }
        
        /// <summary>
         /// Releases the unmanaged memory used by the contour.
         /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (!this.disposedValue)
            {
                if (this.dataPtr != IntPtr.Zero)
                {
                    NativeMethods.cv2ex_FreeContourData(this.dataPtr);
                    this.dataPtr = IntPtr.Zero;
                }
                this.disposedValue = true;
            }
        }

        /// <summary>
        /// Finalizer.
        /// </summary>
        ~Contour() => this.Dispose(disposing: false);

        /// <summary>
        /// Releases the unmanaged memory used by the contour.
        /// </summary>
        public void Dispose()
        {
            this.Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }

    // Internal struct for marshaling from native code
    [StructLayout(LayoutKind.Sequential)]
    internal struct ContourC
    {
        public IntPtr Points;
        public int NumPoints;
        public IntPtr NormalAngles;
        public IntPtr Response;
        public IntPtr IntPoints;
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
                try
                {
                    var contours = new Contour[numContours];
                    var contourCSize = Marshal.SizeOf<ContourC>();

                    for (int i = 0; i < numContours; i++)
                    {
                        IntPtr currentContourPtr = new IntPtr(contoursPtr.ToInt64() + i * contourCSize);
                        var contourC = Marshal.PtrToStructure<ContourC>(currentContourPtr);

                        contours[i] = new Contour(contourC);
                    }
                    return contours;
                }
                finally
                {
                    // Free the C-style array of structs, but not the data within each struct.
                    // The Contour objects now own that memory.
                    NativeMethods.cv2ex_FreeContours(contoursPtr, numContours);
                }
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

            NativeMethods.cv2ex_RefineContourSubPix(
                dxMat.CvPtr, dyMat.CvPtr, initialContour, initialContour.Length, searchRadius, fixCorners,
                out var contourC);

            GC.KeepAlive(dx);
            GC.KeepAlive(dy);
            GC.KeepAlive(dxMat);
            GC.KeepAlive(dyMat);
            return new Contour(contourC);
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
            if (outNumContours > 0 && contoursPtr != IntPtr.Zero) {
                try
                {
                    refinedContours = new Contour[outNumContours];
                    var contourCSize = Marshal.SizeOf<ContourC>();

                    for (int i = 0; i < outNumContours; i++)
                    {
                        IntPtr currentContourPtr = new IntPtr(contoursPtr.ToInt64() + i * contourCSize);
                        var contourC = Marshal.PtrToStructure<ContourC>(currentContourPtr);
                        refinedContours[i] = new Contour(contourC);
                    }
                }
                finally
                {
                    // Free the C-style array of structs, but not the data within each struct.
                    // The Contour objects now own that memory.
                    NativeMethods.cv2ex_FreeContours(contoursPtr, outNumContours);
                }
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
            using (var dx = new Mat())
            using (var dy = new Mat())
            {
                PrecomputeEdgesSubPixBilateral(gray, diameter, sigmaColor, sigmaSpace, gradientAlpha, dx, dy);
                return RefineContourSubPix(dx, dy, initialContours, searchRadius, fixCorners);
            }
        }

        /// <summary>
        /// Precomputes the gradient maps (gradX, gradY) using Sobel operator.
        /// This is useful when refining multiple contours on the same image to avoid redundant computations.
        /// </summary>
        /// <param name="gray">Input 8-bit single-channel image.</param>
        /// <param name="gradX">Output 32-bit float (CV_32F) gradient map in X direction.</param>
        /// <param name="gradY">Output 32-bit float (CV_32F) gradient map in Y direction.</param>
        /// <param name="ksize">Aperture size for the Sobel operator.</param>
        public static void PrecomputeGradientsSobel(InputArray gray, OutputArray gradX, OutputArray gradY, int ksize = 3)
        {
            if (gray == null) throw new ArgumentNullException(nameof(gray));
            if (gradX == null) throw new ArgumentNullException(nameof(gradX));
            if (gradY == null) throw new ArgumentNullException(nameof(gradY));
            gray.ThrowIfDisposed();
            gradX.ThrowIfNotReady();
            gradY.ThrowIfNotReady();

            Mat grayMat = gray.GetMat();
            Mat gradXMat = gradX.GetMat();
            Mat gradYMat = gradY.GetMat();
            NativeMethods.cv2ex_PrecomputeGradientsSobel(grayMat.CvPtr, gradXMat.CvPtr, gradYMat.CvPtr, ksize);

            GC.KeepAlive(gray);
            GC.KeepAlive(gradX);
            GC.KeepAlive(gradY);
            GC.KeepAlive(grayMat);
            GC.KeepAlive(gradXMat);
            GC.KeepAlive(gradYMat);
        }

        /// <summary>
        /// Refines a given integer-precision contour to sub-pixel accuracy using a weighted centroid of gradient magnitudes.
        /// </summary>
        /// <param name="gradX">Precomputed 32-bit float (CV_32F) gradient map in X direction.</param>
        /// <param name="gradY">Precomputed 32-bit float (CV_32F) gradient map in Y direction.</param>
        /// <param name="initialContour">The integer-precision contour to refine.</param>
        /// <param name="windowSize">The size of the window around each point to calculate the centroid.</param>
        /// <returns>The output sub-pixel accurate contour.</returns>
        public static Contour RefineContourCentroid(
            InputArray gradX,
            InputArray gradY,
            Point[] initialContour,
            int windowSize)
        {
            if (gradX == null) throw new ArgumentNullException(nameof(gradX));
            if (gradY == null) throw new ArgumentNullException(nameof(gradY));
            if (initialContour == null) throw new ArgumentNullException(nameof(initialContour));
            gradX.ThrowIfDisposed();
            gradY.ThrowIfDisposed();

            Mat gradXMat = gradX.GetMat();
            Mat gradYMat = gradY.GetMat();

            NativeMethods.cv2ex_RefineContourCentroid(
                gradXMat.CvPtr, gradYMat.CvPtr, initialContour, initialContour.Length, windowSize,
                out var contourC);

            GC.KeepAlive(gradX);
            GC.KeepAlive(gradY);
            GC.KeepAlive(gradXMat);
            GC.KeepAlive(gradYMat);
            return new Contour(contourC);
        }

        /// <summary>
        /// Refines a given integer-precision contour to sub-pixel accuracy using a weighted centroid of gradient magnitudes.
        /// This is a convenience overload that computes Sobel gradients internally.
        /// </summary>
        /// <param name="gray">Input 8-bit single-channel image.</param>
        /// <param name="initialContour">The integer-precision contour to refine.</param>
        /// <param name="windowSize">The size of the window around each point to calculate the centroid.</param>
        /// <returns>The output sub-pixel accurate contour.</returns>
        public static Contour RefineContourCentroid(
            InputArray gray,
            Point[] initialContour,
            int windowSize)
        {
            using (var gradX = new Mat())
            using (var gradY = new Mat())
            {
                PrecomputeGradientsSobel(gray, gradX, gradY);
                return RefineContourCentroid(gradX, gradY, initialContour, windowSize);
            }
        }

        /// <summary>
        /// Refines given integer-precision contours to sub-pixel accuracy using a weighted centroid of gradient magnitudes.
        /// </summary>
        /// <param name="gradX">Precomputed 32-bit float (CV_32F) gradient map in X direction.</param>
        /// <param name="gradY">Precomputed 32-bit float (CV_32F) gradient map in Y direction.</param>
        /// <param name="initialContours">The integer-precision contours to refine.</param>
        /// <param name="windowSize">The size of the window around each point to calculate the centroid.</param>
        /// <returns>The output sub-pixel accurate contours.</returns>
        public static Contour[] RefineContoursCentroid(
            InputArray gradX,
            InputArray gradY,
            System.Collections.Generic.IEnumerable<Point[]> initialContours,
            int windowSize)
        {
            if (gradX == null) throw new ArgumentNullException(nameof(gradX));
            if (gradY == null) throw new ArgumentNullException(nameof(gradY));
            if (initialContours == null) throw new ArgumentNullException(nameof(initialContours));
            gradX.ThrowIfDisposed();
            gradY.ThrowIfDisposed();

            var initialContoursArray = System.Linq.Enumerable.ToArray(initialContours);
            int numContours = initialContoursArray.Length;
            if (numContours == 0) return Array.Empty<Contour>();

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

            Mat gradXMat = gradX.GetMat();
            Mat gradYMat = gradY.GetMat();

            NativeMethods.cv2ex_RefineContoursCentroid(
                gradXMat.CvPtr, gradYMat.CvPtr,
                contoursData, contourLengths, numContours, windowSize,
                out var contoursPtr, out var outNumContours);

            // This part is similar to RefineContourSubPix, could be refactored
            if (outNumContours > 0 && contoursPtr != IntPtr.Zero) {
                try
                {
                    var refinedContours = new Contour[outNumContours];
                    var contourCSize = Marshal.SizeOf<ContourC>();

                    for (int i = 0; i < outNumContours; i++)
                    {
                        IntPtr currentContourPtr = new IntPtr(contoursPtr.ToInt64() + i * contourCSize);
                        var contourC = Marshal.PtrToStructure<ContourC>(currentContourPtr);
                        refinedContours[i] = new Contour(contourC);
                    }
                    return refinedContours;
                }
                finally
                {
                    NativeMethods.cv2ex_FreeContours(contoursPtr, outNumContours);
                }
            }
            return Array.Empty<Contour>();
        }
    }
}
