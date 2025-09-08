namespace OpenCvSharpEx.Sample
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.btnLoadPattern = new System.Windows.Forms.Button();
            this.btnTeach = new System.Windows.Forms.Button();
            this.btnLoadImage = new System.Windows.Forms.Button();
            this.btnSearch = new System.Windows.Forms.Button();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.splitContainer2 = new System.Windows.Forms.SplitContainer();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPageShapeMatcher = new System.Windows.Forms.TabPage();
            this.propertyGridShapeMatcher = new System.Windows.Forms.PropertyGrid();
            this.tabPageRotatedPatternMatcher = new System.Windows.Forms.TabPage();
            this.propertyGridRotatedPatternMatcher = new System.Windows.Forms.PropertyGrid();
            this.txtOutput = new System.Windows.Forms.TextBox();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.tableLayoutPanel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).BeginInit();
            this.splitContainer2.Panel1.SuspendLayout();
            this.splitContainer2.Panel2.SuspendLayout();
            this.splitContainer2.SuspendLayout();
            this.tabControl1.SuspendLayout();
            this.tabPageShapeMatcher.SuspendLayout();
            this.tabPageRotatedPatternMatcher.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.SuspendLayout();
            // 
            // tableLayoutPanel1
            //
            this.tableLayoutPanel1.ColumnCount = 5;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle());
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle());
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle());
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle());
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel1.Controls.Add(this.btnLoadPattern, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.btnTeach, 1, 0);
            this.tableLayoutPanel1.Controls.Add(this.btnLoadImage, 2, 0);
            this.tableLayoutPanel1.Controls.Add(this.btnSearch, 3, 0);
            this.tableLayoutPanel1.Dock = System.Windows.Forms.DockStyle.Top;
            this.tableLayoutPanel1.Location = new System.Drawing.Point(0, 0);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 1;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(984, 40);
            this.tableLayoutPanel1.TabIndex = 0;
            //
            // btnLoadPattern
            //
            this.btnLoadPattern.AutoSize = true;
            this.btnLoadPattern.Location = new System.Drawing.Point(3, 3);
            this.btnLoadPattern.Name = "btnLoadPattern";
            this.btnLoadPattern.Size = new System.Drawing.Size(100, 30);
            this.btnLoadPattern.TabIndex = 0;
            this.btnLoadPattern.Text = "Load Pattern";
            this.btnLoadPattern.UseVisualStyleBackColor = true;
            this.btnLoadPattern.Click += new System.EventHandler(this.OnLoadPatternClick);
            //
            // btnTeach
            //
            this.btnTeach.AutoSize = true;
            this.btnTeach.Location = new System.Drawing.Point(109, 3);
            this.btnTeach.Name = "btnTeach";
            this.btnTeach.Size = new System.Drawing.Size(100, 30);
            this.btnTeach.TabIndex = 1;
            this.btnTeach.Text = "Teach";
            this.btnTeach.UseVisualStyleBackColor = true;
            this.btnTeach.Click += new System.EventHandler(this.OnTeachClick);
            //
            // btnLoadImage
            //
            this.btnLoadImage.AutoSize = true;
            this.btnLoadImage.Location = new System.Drawing.Point(215, 3);
            this.btnLoadImage.Name = "btnLoadImage";
            this.btnLoadImage.Size = new System.Drawing.Size(100, 30);
            this.btnLoadImage.TabIndex = 2;
            this.btnLoadImage.Text = "Load Image";
            this.btnLoadImage.UseVisualStyleBackColor = true;
            this.btnLoadImage.Click += new System.EventHandler(this.OnLoadImageClick);
            //
            // btnSearch
            //
            this.btnSearch.AutoSize = true;
            this.btnSearch.Location = new System.Drawing.Point(321, 3);
            this.btnSearch.Name = "btnSearch";
            this.btnSearch.Size = new System.Drawing.Size(100, 30);
            this.btnSearch.TabIndex = 3;
            this.btnSearch.Text = "Search";
            this.btnSearch.UseVisualStyleBackColor = true;
            this.btnSearch.Click += new System.EventHandler(this.OnSearchClick);
            //
            // splitContainer1
            //
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(0, 40);
            this.splitContainer1.Name = "splitContainer1";
            //
            // splitContainer1.Panel1
            //
            this.splitContainer1.Panel1.Controls.Add(this.splitContainer2);
            //
            // splitContainer1.Panel2
            //
            this.splitContainer1.Panel2.Controls.Add(this.pictureBox1);
            this.splitContainer1.Size = new System.Drawing.Size(984, 521);
            this.splitContainer1.SplitterDistance = 328;
            this.splitContainer1.TabIndex = 1;
            //
            // splitContainer2
            //
            this.splitContainer2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer2.Location = new System.Drawing.Point(0, 0);
            this.splitContainer2.Name = "splitContainer2";
            this.splitContainer2.Orientation = System.Windows.Forms.Orientation.Horizontal;
            // 
            // splitContainer2.Panel1
            // 
            this.splitContainer2.Panel1.Controls.Add(this.tabControl1);
            // 
            // splitContainer2.Panel2
            // 
            this.splitContainer2.Panel2.Controls.Add(this.txtOutput);
            this.splitContainer2.Size = new System.Drawing.Size(328, 521);
            this.splitContainer2.SplitterDistance = 280;
            this.splitContainer2.TabIndex = 0;
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPageShapeMatcher);
            this.tabControl1.Controls.Add(this.tabPageRotatedPatternMatcher);
            this.tabControl1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tabControl1.Location = new System.Drawing.Point(0, 0);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(328, 280);
            this.tabControl1.TabIndex = 1;
            // 
            // tabPageShapeMatcher
            // 
            this.tabPageShapeMatcher.Controls.Add(this.propertyGridShapeMatcher);
            this.tabPageShapeMatcher.Location = new System.Drawing.Point(4, 25);
            this.tabPageShapeMatcher.Name = "tabPageShapeMatcher";
            this.tabPageShapeMatcher.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageShapeMatcher.Size = new System.Drawing.Size(320, 251);
            this.tabPageShapeMatcher.TabIndex = 0;
            this.tabPageShapeMatcher.Text = "Shape Matcher";
            this.tabPageShapeMatcher.UseVisualStyleBackColor = true;
            // 
            // propertyGridShapeMatcher
            // 
            this.propertyGridShapeMatcher.Dock = System.Windows.Forms.DockStyle.Fill;
            this.propertyGridShapeMatcher.Location = new System.Drawing.Point(3, 3);
            this.propertyGridShapeMatcher.Name = "propertyGridShapeMatcher";
            this.propertyGridShapeMatcher.Size = new System.Drawing.Size(314, 245);
            this.propertyGridShapeMatcher.TabIndex = 0;
            this.propertyGridShapeMatcher.PropertyValueChanged += new System.Windows.Forms.PropertyValueChangedEventHandler(this.OnPropertyValueChanged);
            // 
            // tabPageRotatedPatternMatcher
            // 
            this.tabPageRotatedPatternMatcher.Controls.Add(this.propertyGridRotatedPatternMatcher);
            this.tabPageRotatedPatternMatcher.Location = new System.Drawing.Point(4, 25);
            this.tabPageRotatedPatternMatcher.Name = "tabPageRotatedPatternMatcher";
            this.tabPageRotatedPatternMatcher.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageRotatedPatternMatcher.Size = new System.Drawing.Size(320, 251);
            this.tabPageRotatedPatternMatcher.TabIndex = 1;
            this.tabPageRotatedPatternMatcher.Text = "Rotated Pattern Matcher";
            this.tabPageRotatedPatternMatcher.UseVisualStyleBackColor = true;
            // 
            // propertyGridRotatedPatternMatcher
            // 
            this.propertyGridRotatedPatternMatcher.Dock = System.Windows.Forms.DockStyle.Fill;
            this.propertyGridRotatedPatternMatcher.Location = new System.Drawing.Point(3, 3);
            this.propertyGridRotatedPatternMatcher.Name = "propertyGridRotatedPatternMatcher";
            this.propertyGridRotatedPatternMatcher.Size = new System.Drawing.Size(314, 245);
            this.propertyGridRotatedPatternMatcher.TabIndex = 0;
            this.propertyGridRotatedPatternMatcher.PropertyValueChanged += new System.Windows.Forms.PropertyValueChangedEventHandler(this.OnRotatedPatternMatcherPropertyValueChanged);
            // 
            // txtOutput
            // 
            this.txtOutput.Dock = System.Windows.Forms.DockStyle.Fill;
            this.txtOutput.Location = new System.Drawing.Point(0, 0);
            this.txtOutput.Multiline = true;
            this.txtOutput.Name = "txtOutput";
            this.txtOutput.ReadOnly = true;
            this.txtOutput.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.txtOutput.Size = new System.Drawing.Size(328, 237);
            this.txtOutput.TabIndex = 0;
            //
            // pictureBox1
            //
            this.pictureBox1.BackColor = System.Drawing.Color.Black;
            this.pictureBox1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pictureBox1.Location = new System.Drawing.Point(0, 0);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(652, 521);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox1.TabIndex = 0;
            this.pictureBox1.TabStop = false;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(984, 561);
            this.Controls.Add(this.splitContainer1);
            this.Controls.Add(this.tableLayoutPanel1);
            this.Name = "Form1";
            this.Text = "OpenCvSharpEx ShapeMatcher Sample";
            this.tableLayoutPanel1.ResumeLayout(false);
            this.tableLayoutPanel1.PerformLayout();
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.splitContainer2.Panel1.ResumeLayout(false);
            this.splitContainer2.Panel2.ResumeLayout(false);
            this.splitContainer2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).EndInit();
            this.splitContainer2.ResumeLayout(false);
            this.tabControl1.ResumeLayout(false);
            this.tabPageShapeMatcher.ResumeLayout(false);
            this.tabPageRotatedPatternMatcher.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
        private System.Windows.Forms.Button btnLoadPattern;
        private System.Windows.Forms.Button btnTeach;
        private System.Windows.Forms.Button btnLoadImage;
        private System.Windows.Forms.Button btnSearch;
        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.SplitContainer splitContainer2;
        private System.Windows.Forms.PropertyGrid propertyGridShapeMatcher;
        private System.Windows.Forms.TextBox txtOutput;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage tabPageShapeMatcher;
        private System.Windows.Forms.TabPage tabPageRotatedPatternMatcher;
        private System.Windows.Forms.PropertyGrid propertyGridRotatedPatternMatcher;
    }
}

