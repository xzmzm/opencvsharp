using OpenCvSharp;
using System;

namespace OpenCvSharpEx
{
    /// <summary>
    /// Provides extension methods for OpenCvSharp types.
    /// </summary>
    public static class RotatedRectExtensions
    {
        /// <summary>
        /// Calculates the up-right bounding rectangle of the rotated rectangle with floating-point precision.
        /// </summary>
        /// <param name="rotatedRect">The rotated rectangle.</param>
        /// <returns>A Rect2d representing the bounding rectangle.</returns>
        public static Rect2d BoundingRect2d(this RotatedRect rotatedRect)
        {
            Point2f[] points = rotatedRect.Points();
            if (points == null || points.Length != 4)
            {
                return new Rect2d(); // Return empty rect if points are not valid
            }

            double minX = points[0].X, maxX = points[0].X;
            double minY = points[0].Y, maxY = points[0].Y;

            for (int i = 1; i < 4; i++)
            {
                minX = Math.Min(minX, points[i].X);
                maxX = Math.Max(maxX, points[i].X);
                minY = Math.Min(minY, points[i].Y);
                maxY = Math.Max(maxY, points[i].Y);
            }

            return new Rect2d(minX, minY, maxX - minX, maxY - minY);
        }
    }
}
