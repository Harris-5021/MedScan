using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Authorization;

namespace WebApplication1.Controllers
{
    [Authorize]
    public class DashboardController : Controller
    {
        private readonly IWebHostEnvironment _webHostEnvironment;

        public DashboardController(IWebHostEnvironment webHostEnvironment)
        {
            _webHostEnvironment = webHostEnvironment;
        }

        // GET: /Dashboard/UserDashboard
        public IActionResult UserDashboard()
        {
            return View();
        }

        // POST: /Dashboard/UploadXRay
        [HttpPost]
        public async Task<IActionResult> UploadXRay(IFormFile xrayImage)
        {
            if (xrayImage != null && xrayImage.Length > 0)
            {
                try
                {
                    // Generate a unique file name
                    var fileName = Path.GetFileNameWithoutExtension(xrayImage.FileName);
                    var extension = Path.GetExtension(xrayImage.FileName);
                    var newFileName = $"{fileName}_{Guid.NewGuid()}{extension}";

                    // Path to save the uploaded file
                    var uploadsFolder = Path.Combine(_webHostEnvironment.WebRootPath, "uploads");
                    Directory.CreateDirectory(uploadsFolder);
                    var filePath = Path.Combine(uploadsFolder, newFileName);

                    // Save the uploaded file
                    using (var stream = new FileStream(filePath, FileMode.Create))
                    {
                        await xrayImage.CopyToAsync(stream);
                    }

                    // Pass data to the view
                    ViewBag.ImagePath = $"/uploads/{newFileName}";
                }
                catch (Exception ex)
                {
                    ViewBag.Error = $"Error uploading file: {ex.Message}";
                }
            }
            else
            {
                ViewBag.Error = "No file selected.";
            }

            return View("UserDashboard");
        }
    }
}
