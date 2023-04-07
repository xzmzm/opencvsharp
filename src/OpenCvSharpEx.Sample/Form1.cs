using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
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
            var patternFile = @"D:\src\vision\Fastest_Image_Pattern_Matching\Test Images\20220611.bmp";
            this.shapeMatcher = new ShapeMatcher();
            using (var pattern = Cv2.ImRead(patternFile))
            {
                if (pattern.Channels() > 1)
                    Cv2.CvtColor(pattern, pattern, ColorConversionCodes.RGB2GRAY);
                this.shapeMatcher.Teach(pattern);
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            var imageFile = @"D:\src\vision\Fastest_Image_Pattern_Matching\Test Images\Src1.bmp";
            using (var image = Cv2.ImRead(imageFile))
            {
                if (image.Channels() > 1)
                    Cv2.CvtColor(image, image, ColorConversionCodes.RGB2GRAY);
                var r = this.shapeMatcher.Search(image);
            }

        }
    }
}
