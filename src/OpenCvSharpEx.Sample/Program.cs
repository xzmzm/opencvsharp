using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace OpenCvSharpEx.Sample
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            var additionalPathDirs = new[]
{
                @"Q:\src\vision\opencvsharp\src\Release\x64",
                @"D:\src\vision\opencvsharp\src\Debug1\x64"
            };

            void AddDirectoriesToPath(string[] additionalPathDirs1)
            {
                var path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
                foreach (var dir in additionalPathDirs1)
                {
                    if (Directory.Exists(dir) && !path.Contains(dir))
                    {
                        path = dir + ";" + path;
                    }
                }
                Environment.SetEnvironmentVariable("PATH", path);
            }
            AddDirectoriesToPath(additionalPathDirs);

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool SetDllDirectory(string lpPathName);
    }
}
