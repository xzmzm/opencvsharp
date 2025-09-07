using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace OpenCvSharpEx.Sample
{
    public partial class Form1 : Form
    {
        private ShapeMatcher shapeMatcher;
        private Mat patternMat;
        private Mat searchImageMat;
        private ShapeMatcherResults lastSearchResult;

        public Form1()
        {
            this.InitializeComponent();
            this.shapeMatcher = new ShapeMatcher() { MinAngle = 0, MaxAngle = 360, UseFusion = false, Refinement = RefinementMethod.FastQuadratic };
            this.propertyGrid1.SelectedObject = this.shapeMatcher;
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

            using (var gray = new Mat())
            {
                if (this.patternMat.Channels() > 1)
                    Cv2.CvtColor(this.patternMat, gray, ColorConversionCodes.BGR2GRAY);
                else
                    this.patternMat.CopyTo(gray);

                this.Log("Teaching pattern...");
                var sw = Stopwatch.StartNew();
                this.shapeMatcher.Teach(gray);
                sw.Stop();
                this.Log($"Teaching complete in {sw.ElapsedMilliseconds} ms.");

                this.DrawFeatures();
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
                    this.Log($"Search image '{ofd.FileName}' loaded.");
                }
            }
        }

        private void OnSearchClick(object sender, EventArgs e)
        {
            if (this.shapeMatcher == null)
            {
                MessageBox.Show("Matcher not initialized. Please teach a pattern first.");
                return;
            }
            if (this.searchImageMat == null)
            {
                MessageBox.Show("Please load an image to search in.");
                return;
            }

            using (var gray = new Mat())
            {
                if (this.searchImageMat.Channels() > 1)
                    Cv2.CvtColor(this.searchImageMat, gray, ColorConversionCodes.BGR2GRAY);
                else
                    this.searchImageMat.CopyTo(gray);

                this.Log("Searching...");
                var sw = Stopwatch.StartNew();
                try
                {
                    this.lastSearchResult = this.shapeMatcher.Search(gray);
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
            if (this.searchImageMat == null || this.lastSearchResult == null) return;

            using (var resultMat = new Mat())
            {
                if (this.searchImageMat.Channels() == 1)
                    Cv2.CvtColor(this.searchImageMat, resultMat, ColorConversionCodes.GRAY2BGR);
                else
                    this.searchImageMat.CopyTo(resultMat);

                var points = this.lastSearchResult.RotatedBounds.Points().Select(p => new OpenCvSharp.Point(p.X, p.Y)).ToArray();
                Cv2.Polylines(resultMat, new[] { points }, true, Scalar.LimeGreen, 2);

                var center = this.lastSearchResult.Location;
                Cv2.DrawMarker(resultMat, new OpenCvSharp.Point(center.X, center.Y), Scalar.Red, MarkerTypes.Cross, 10, 2);

                this.pictureBox1.Image?.Dispose();
                this.pictureBox1.Image = resultMat.ToBitmap();
            }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);
            this.shapeMatcher?.Dispose();
            this.patternMat?.Dispose();
            this.searchImageMat?.Dispose();
            this.pictureBox1.Image?.Dispose();
        }

        private void OnPropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            if (this.shapeMatcher != null)
            {
                this.Log($"Matcher property changed: {e.ChangedItem.Label} = {e.ChangedItem.Value}");
            }
        }
    }
}
