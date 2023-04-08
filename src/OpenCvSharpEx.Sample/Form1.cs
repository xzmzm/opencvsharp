﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenCvSharp;

namespace OpenCvSharpEx.Sample
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
        ShapeMatcher shapeMatcher;
        private void button1_Click(object sender, EventArgs e)
        {
            var patternFile = @"Q:\src\vision\Fastest_Image_Pattern_Matching\Test Images\20220611.bmp";
            patternFile = @"Q:\src\vision\shape_based_matching\test\case1\train.png";
            this.shapeMatcher = new ShapeMatcher() { MinAngle = 0, MaxAngle = 360 };
            using (var pattern = Cv2.ImRead(patternFile))
            {
                using (var mat = new Mat(pattern, new Rect(50, 50, pattern.Cols - 100, pattern.Rows - 100)))
                {
                    if (mat.Channels() > 1)
                        Cv2.CvtColor(mat, mat, ColorConversionCodes.RGB2GRAY);
                    this.shapeMatcher.Teach(mat);
                }
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            var imageFile = @"Q:\src\vision\Fastest_Image_Pattern_Matching\Test Images\Src1.bmp";
            imageFile = @"Q:\src\vision\shape_based_matching\test\case1\test.png";
            using (var image = Cv2.ImRead(imageFile))
            {
                if (image.Channels() > 1)
                    Cv2.CvtColor(image, image, ColorConversionCodes.RGB2GRAY);
                var sw = Stopwatch.StartNew();
                var r = this.shapeMatcher.Search(image);
                Debug.WriteLine($"Time: {sw.ElapsedMilliseconds} ms");
            }

        }
    }
}
