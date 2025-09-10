using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using OpenCvSharpEx;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System.ComponentModel;
using System.Collections.Generic;

namespace OpenCvSharpEx.Sample
{
    public class EdgesSubPixSettings
    {
        [Description("The alpha parameter for the Gaussian filter (sigma).")]
        public double Alpha { get; set; } = 1.0;

        [Description("The lower hysteresis threshold.")]
        public int LowThreshold { get; set; } = 50;

        [Description("The higher hysteresis threshold.")]
        public int HighThreshold { get; set; } = 100;

        [Description("Contour retrieval mode.")]
        public RetrievalModes RetrievalMode { get; set; } = RetrievalModes.List;

        [Description("If true, finds binary contours using BinaryThreshold and then refines them.")]
        public bool RefineBinaryContours { get; set; } = false;

        [Description("Threshold for binary contour detection (0-255). Used when RefineBinaryContours is true.")]
        public int BinaryThreshold { get; set; } = 128;

        [Description("Search radius in pixels for sub-pixel refinement. Used when RefineBinaryContours is true.")]
        public int SearchRadius { get; set; } = 3;

        [Description("Enables an advanced corner-finding algorithm for refined contours. This fits lines to adjacent segments to find a precise corner, preventing inward distortion.")]
        public bool FixCorners { get; set; } = true;
    }

    public class RotatedPatternMatcherSettings
    {
        [Description("The minimum score for a match to be considered valid (0-100).")]
        public double AcceptanceScore { get; set; } = 90.0;

        [Description("The minimum rotation angle to search for in degrees.")]
        public double MinAngle { get; set; } = -180.0;

        [Description("The maximum rotation angle to search for in degrees.")]
        public double MaxAngle { get; set; } = 180.0;

        [Description("The step size for angle search in degrees.")]
        public double AngleStep { get; set; } = 1.0;

        [Description("The maximum number of matches to find.")]
        public int MaxMatchCount { get; set; } = 10;

        [Description("The maximum allowed overlap ratio between found matches (0-1). A value of 0 means no overlap is allowed.")]
        public double MaxOverlapRatio { get; set; } = 0.0;

        [Description("The number of pyramid levels to use for matching. Higher values are faster but less accurate for small patterns.")]
        public int PyramidLevels { get; set; } = 4;
    }

    public partial class Form1 : Form
    {
        private ShapeMatcher shapeMatcher;
        private RotatedPatternMatcher rotatedPatternMatcher;
        private RotatedPatternMatcherSettings rotatedPatternMatcherSettings;
        private EdgesSubPixSettings edgesSubPixSettings;
        private Mat patternMat;
        private Mat searchImageMat;
        private ShapeMatcherResults lastSearchResult;
        private RotationPatternMatcherResults[] lastRotatedSearchResult;
        private Contour[] lastEdgesSubPixResult;

        public Form1()
        {
            this.InitializeComponent();
            this.shapeMatcher = new ShapeMatcher() { MinAngle = 0, MaxAngle = 360, UseFusion = false, Refinement = RefinementMethod.FastQuadratic };
            this.propertyGridShapeMatcher.SelectedObject = this.shapeMatcher;

            this.rotatedPatternMatcher = new RotatedPatternMatcher();
            this.rotatedPatternMatcherSettings = new RotatedPatternMatcherSettings();
            this.propertyGridRotatedPatternMatcher.SelectedObject = this.rotatedPatternMatcherSettings;

            this.edgesSubPixSettings = new EdgesSubPixSettings();
            this.propertyGridEdgesSubPix.SelectedObject = this.edgesSubPixSettings;
        }

        private void OnLoadPatternClick(object sender, EventArgs e)
        {
            // Q:\src\vision\shape_based_matching\test\case1\train.png
            using (var ofd = new OpenFileDialog())
            {
                ofd.Filter = "Image Files|*.bmp;*.png;*.jpg;*.jpeg|All files (*.*)|*.*";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    this.patternMat?.Dispose();
                    this.patternMat = new Mat(ofd.FileName, ImreadModes.Color);
                    if (this.patternMat.Empty())
                    {
                        MessageBox.Show("Could not open or find the image");
                        this.patternMat = null;
                        return;
                    }
                    this.pictureBox1.Image?.Dispose();
                    this.pictureBox1.Image = this.patternMat.ToBitmap();
                    this.Log($"Pattern '{ofd.FileName}' loaded.");
                }
            }
        }

        private void OnTeachClick(object sender, EventArgs e)
        {
            if (this.patternMat == null)
            {
                MessageBox.Show("Please load a pattern image first.");
                return;
            }

            if (this.tabControl1.SelectedIndex == 0) // Shape Matcher
            {
                using (var gray = new Mat())
                {
                    if (this.patternMat.Channels() > 1)
                        Cv2.CvtColor(this.patternMat, gray, ColorConversionCodes.BGR2GRAY);
                    else
                        this.patternMat.CopyTo(gray);

                    this.Log("Teaching pattern for Shape Matcher...");
                    var sw = Stopwatch.StartNew();
                    this.shapeMatcher.Teach(gray);
                    sw.Stop();
                    this.Log($"Teaching complete in {sw.ElapsedMilliseconds} ms.");

                    this.DrawFeatures();
                }
            }
            else // Rotated Pattern Matcher
            {
                using (var gray = new Mat())
                {
                    if (this.patternMat.Channels() > 1)
                        Cv2.CvtColor(this.patternMat, gray, ColorConversionCodes.BGR2GRAY);
                    else
                        this.patternMat.CopyTo(gray);

                    this.Log("Teaching pattern for Rotated Pattern Matcher...");
                    var sw = Stopwatch.StartNew();
                    this.rotatedPatternMatcher.Teach(gray, this.rotatedPatternMatcherSettings.PyramidLevels);
                    sw.Stop();
                    this.Log($"Teaching complete in {sw.ElapsedMilliseconds} ms.");
                }
            }
        }

        private void OnLoadImageClick(object sender, EventArgs e)
        {
            // Q:\src\vision\shape_based_matching\test\case1\test.png
            using (var ofd = new OpenFileDialog())
            {
                ofd.Filter = "Image Files|*.bmp;*.png;*.jpg;*.jpeg|All files (*.*)|*.*";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    this.searchImageMat?.Dispose();
                    this.searchImageMat = new Mat(ofd.FileName, ImreadModes.Color);
                    if (this.searchImageMat.Empty())
                    {
                        MessageBox.Show("Could not open or find the image");
                        this.searchImageMat = null;
                        return;
                    }
                    this.pictureBox1.Image?.Dispose();
                    this.pictureBox1.Image = this.searchImageMat.ToBitmap();
                    this.lastSearchResult = null;
                    this.lastRotatedSearchResult = null;
                    this.lastEdgesSubPixResult = null;
                    this.Log($"Search image '{ofd.FileName}' loaded.");
                }
            }
        }

        private void OnSearchClick(object sender, EventArgs e)
        {
            if (this.searchImageMat == null)
            {
                MessageBox.Show("Please load an image to process.");
                return;
            }

            using (var gray = new Mat())
            {
                if (this.searchImageMat.Channels() > 1)
                    Cv2.CvtColor(this.searchImageMat, gray, ColorConversionCodes.BGR2GRAY);
                else
                    this.searchImageMat.CopyTo(gray);

                this.Log("Processing...");
                var sw = Stopwatch.StartNew();
                if (this.tabControl1.SelectedIndex == 0) // Shape Matcher
                {
                    try
                    {
                        this.Log("Searching...");
                        this.lastSearchResult = this.shapeMatcher.Search(gray);
                        sw.Stop();

                        if (this.lastSearchResult != null && this.lastSearchResult.Score > 0)
                        {
                            this.Log($"Search complete in {sw.ElapsedMilliseconds} ms.");
                            this.Log($"Result:");
                            this.Log($"  Score: {this.lastSearchResult.Score:F2}");
                            this.Log($"  Angle: {this.lastSearchResult.Angle:F2}°");
                            this.Log($"  Location: ({this.lastSearchResult.Location.X:F2}, {this.lastSearchResult.Location.Y:F2})");
                            this.Log($"  Template ID: {this.lastSearchResult.TemplateID}");

                            this.DrawSearchResult();
                        }
                        else
                        {
                            this.Log($"Search complete in {sw.ElapsedMilliseconds} ms. No match found.");
                            this.pictureBox1.Image?.Dispose();
                            this.pictureBox1.Image = this.searchImageMat.ToBitmap();
                        }
                    }
                    catch (OpenCvSharpException ex)
                    {
                        this.Log($"Error during search: {ex.Message}");
                        if (ex.Message.Contains("No pattern is taught yet"))
                        {
                            MessageBox.Show("Please teach a pattern before searching.");
                        }
                        return;
                    }
                }
                else if (this.tabControl1.SelectedIndex == 1) // Rotated Pattern Matcher
                {
                    try
                    {
                        this.Log("Searching...");
                        this.lastRotatedSearchResult = this.rotatedPatternMatcher.Search(
                            gray,
                            this.rotatedPatternMatcherSettings.AcceptanceScore,
                            this.rotatedPatternMatcherSettings.MinAngle,
                            this.rotatedPatternMatcherSettings.MaxAngle,
                            this.rotatedPatternMatcherSettings.AngleStep,
                            this.rotatedPatternMatcherSettings.MaxMatchCount,
                            this.rotatedPatternMatcherSettings.MaxOverlapRatio);
                        sw.Stop();

                        if (this.lastRotatedSearchResult != null && this.lastRotatedSearchResult.Length > 0)
                        {
                            this.Log($"Search complete in {sw.ElapsedMilliseconds} ms. Found {this.lastRotatedSearchResult.Length} matches.");
                            for (int i = 0; i < this.lastRotatedSearchResult.Length; i++)
                            {
                                var result = this.lastRotatedSearchResult[i];
                                this.Log($"  Match {i + 1}: Score={result.Score:F2}, Angle={result.Angle:F2}°, Location=({result.Location.X:F2}, {result.Location.Y:F2})");
                            }
                            this.DrawSearchResult();
                        }
                        else
                        {
                            this.Log($"Search complete in {sw.ElapsedMilliseconds} ms. No match found.");
                            this.pictureBox1.Image?.Dispose();
                            this.pictureBox1.Image = this.searchImageMat.ToBitmap();
                        }
                    }
                    catch (OpenCvSharpException ex)
                    {
                        this.Log($"Error during search: {ex.Message}");
                    }
                }
                else // Edges SubPix
                {
                    try
                    {
                        if (!this.edgesSubPixSettings.RefineBinaryContours)
                        {
                            this.Log("Finding sub-pixel edges directly...");
                            this.lastEdgesSubPixResult = Cv2Ex.EdgesSubPix(
                                gray,
                                this.edgesSubPixSettings.Alpha,
                                this.edgesSubPixSettings.LowThreshold,
                                this.edgesSubPixSettings.HighThreshold,
                                null, // no hierarchy for now
                                this.edgesSubPixSettings.RetrievalMode);
                            sw.Stop();

                            if (this.lastEdgesSubPixResult != null && this.lastEdgesSubPixResult.Length > 0)
                            {
                                this.Log($"Edge detection complete in {sw.ElapsedMilliseconds} ms. Found {this.lastEdgesSubPixResult.Length} contours.");
                                this.DrawEdgesSubPixResult();
                            }
                            else
                            {
                                this.Log($"Edge detection complete in {sw.ElapsedMilliseconds} ms. No contours found.");
                                this.pictureBox1.Image?.Dispose();
                                this.pictureBox1.Image = this.searchImageMat.ToBitmap();
                            }
                        }
                        else
                        {
                            this.Log("Refining binary contours...");

                            // 1. Find binary contours
                            using (var binary = new Mat())
                            {
                                Cv2.Threshold(gray, binary, this.edgesSubPixSettings.BinaryThreshold, 255, ThresholdTypes.Binary);
                                Cv2.FindContours(binary, out var binaryContours, out _, this.edgesSubPixSettings.RetrievalMode, ContourApproximationModes.ApproxNone);
                                this.Log($"Found {binaryContours.Length} binary contours.");

                                // 2. Refine them
                                var refinedContours = new List<Contour>();
                                using (var dx = new Mat())
                                using (var dy = new Mat())
                                {
                                    Cv2Ex.PrecomputeEdgesSubPix(gray, this.edgesSubPixSettings.Alpha, dx, dy);
                                    foreach (var initialContour in binaryContours)
                                    {
                                        if (initialContour.Length < 3) continue; // Point[]

                                        // RefineContourSubPix now returns a disposable object
                                        var refinedContour = Cv2Ex.RefineContourSubPix(dx, dy, initialContour, this.edgesSubPixSettings.SearchRadius, this.edgesSubPixSettings.FixCorners);
                                        if (!refinedContour.IsEmpty)
                                        {
                                            refinedContours.Add(refinedContour);
                                        }
                                        else
                                        {
                                            // Dispose immediately if not being stored
                                            refinedContour.Dispose();
                                        }
                                    }
                                }

                                // Dispose previous results before assigning new ones
                                this.DisposeLastEdgesSubPixResult();

                                this.lastEdgesSubPixResult = refinedContours.ToArray();
                                sw.Stop();
                                this.Log($"Refinement complete in {sw.ElapsedMilliseconds} ms. Refined {this.lastEdgesSubPixResult.Length} contours.");
                                this.DrawEdgesSubPixResult(binaryContours);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        this.Log($"Error during edge detection: {ex.Message}");
                        this.lastEdgesSubPixResult = null;
                        this.DisposeLastEdgesSubPixResult();
                    }
                }
            }
        }

        private void Log(string message)
        {
            if (this.txtOutput.InvokeRequired)
            {
                this.txtOutput.Invoke(new Action(() => this.Log(message)));
            }
            else
            {
                this.txtOutput.AppendText(message + Environment.NewLine);
            }
        }

        private void DrawFeatures()
        {
            if (this.patternMat == null || this.shapeMatcher == null) return;

            var templateInfo = this.shapeMatcher.GetTemplate(0);
            if (templateInfo == null || templateInfo.Features == null) return;

            var features = templateInfo.Features;
            var bmp = this.patternMat.ToBitmap();
            using (var g = Graphics.FromImage(bmp))
            {
                var offset = this.shapeMatcher.GetPatternOffset();
                foreach (var f in features)
                    g.DrawEllipse(Pens.Lime, f.x - offset.X - 1, f.y - offset.Y - 1, 3, 3);
            }
            this.pictureBox1.Image?.Dispose();
            this.pictureBox1.Image = bmp;
        }

        private void DrawSearchResult()
        {
            if (this.searchImageMat == null) return;

            using (var resultMat = new Mat())
            {
                if (this.searchImageMat.Channels() == 1)
                    Cv2.CvtColor(this.searchImageMat, resultMat, ColorConversionCodes.GRAY2BGR);
                else
                    this.searchImageMat.CopyTo(resultMat);

                if (this.tabControl1.SelectedIndex == 0 && this.lastSearchResult != null) // Shape Matcher
                {
                    var points = this.lastSearchResult.RotatedBounds.Points().Select(p => new OpenCvSharp.Point(p.X, p.Y)).ToArray();
                    Cv2.Polylines(resultMat, new[] { points }, true, Scalar.LimeGreen, 2);

                    var center = this.lastSearchResult.Location;
                    Cv2.DrawMarker(resultMat, new OpenCvSharp.Point(center.X, center.Y), Scalar.Red, MarkerTypes.Cross, 10, 2);
                }
                else if (this.tabControl1.SelectedIndex == 1 && this.lastRotatedSearchResult != null) // Rotated Pattern Matcher
                {
                    foreach (var result in this.lastRotatedSearchResult)
                    {
                        var points = result.RotatedBounds.Points().Select(p => new OpenCvSharp.Point(p.X, p.Y)).ToArray();
                        Cv2.Polylines(resultMat, new[] { points }, true, Scalar.LimeGreen, 2);

                        var center = result.Location;
                        Cv2.DrawMarker(resultMat, new OpenCvSharp.Point(center.X, center.Y), Scalar.Red, MarkerTypes.Cross, 10, 2);
                    }
                }

                this.pictureBox1.Image?.Dispose();
                this.pictureBox1.Image = resultMat.ToBitmap();
            }
        }

        private void DrawEdgesSubPixResult(OpenCvSharp.Point[][] initialContours = null)
        {
            if (this.searchImageMat == null || this.lastEdgesSubPixResult == null) return;

            using (var resultMat = new Mat())
            {
                if (this.searchImageMat.Channels() == 1)
                    Cv2.CvtColor(this.searchImageMat, resultMat, ColorConversionCodes.GRAY2BGR);
                else
                    this.searchImageMat.CopyTo(resultMat);

                // Draw initial contours if provided (e.g., from binary threshold)
                if (initialContours != null)
                {
                    Cv2.DrawContours(resultMat, initialContours, -1, Scalar.Yellow, 1);
                }

                var rng = new Random();
                foreach (var contour in this.lastEdgesSubPixResult)
                {
                    if (contour.Length < 2) continue; // This is a Contour object

                    var color = new Scalar(rng.Next(0, 256), rng.Next(0, 256), rng.Next(0, 256));
                    var pointsSpan = contour.GetPoints();
                    var points = new OpenCvSharp.Point[pointsSpan.Length];
                    for (int i = 0; i < pointsSpan.Length; i++)
                    {
                        points[i] = (OpenCvSharp.Point)pointsSpan[i];
                    }

                    Cv2.Polylines(resultMat, new[] { points }, false, color, 1, LineTypes.AntiAlias);
                }

                this.pictureBox1.Image?.Dispose();
                this.pictureBox1.Image = resultMat.ToBitmap();
            }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);
            this.shapeMatcher?.Dispose();
            this.rotatedPatternMatcher?.Dispose();
            this.patternMat?.Dispose();
            this.searchImageMat?.Dispose();
            this.DisposeLastEdgesSubPixResult();
            this.pictureBox1.Image?.Dispose();
        }

        private void DisposeLastEdgesSubPixResult()
        {
            if (this.lastEdgesSubPixResult != null)
            {
                foreach (var contour in this.lastEdgesSubPixResult)
                {
                    contour.Dispose();
                }
                this.lastEdgesSubPixResult = null;
            }
        }

        private void OnPropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            if (this.shapeMatcher != null)
            {
                this.Log($"Matcher property changed: {e.ChangedItem.Label} = {e.ChangedItem.Value}");
            }
        }

        private void OnRotatedPatternMatcherPropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            if (this.rotatedPatternMatcherSettings != null)
            {
                this.Log($"Rotated Pattern Matcher property changed: {e.ChangedItem.Label} = {e.ChangedItem.Value}");
            }
        }

        private void OnEdgesSubPixPropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            if (this.edgesSubPixSettings != null)
            {
                this.Log($"EdgesSubPix property changed: {e.ChangedItem.Label} = {e.ChangedItem.Value}");
            }
        }

        private void OnTabControlSelectedIndexChanged(object sender, EventArgs e)
        {
            bool isMatcherTab = this.tabControl1.SelectedIndex == 0 || this.tabControl1.SelectedIndex == 1;
            this.btnLoadPattern.Visible = isMatcherTab;
            this.btnTeach.Visible = isMatcherTab;

            if (this.tabControl1.SelectedIndex == 2) // Edges SubPix
            {
                this.btnSearch.Text = "Find Edges";
                if (this.patternMat != null && this.searchImageMat == null)
                {
                    this.searchImageMat = this.patternMat.Clone();
                    this.Log("Using pattern image as source for edge detection.");
                    this.pictureBox1.Image?.Dispose();
                    this.pictureBox1.Image = this.searchImageMat.ToBitmap();
                    this.DisposeLastEdgesSubPixResult();
                    this.lastEdgesSubPixResult = null;
                }
            }
            else
            {
                this.btnSearch.Text = "Search";
            }
        }
    }
}
