import os
from PIL import Image

def resize_images_in_directory(directory_path, target_size=(512, 512)):
    """
    Resizes all PNG images in the specified directory to the target size.

    :param directory_path: The path to the directory containing the PNG images.
    :param target_size: A tuple specifying the desired image size (width, height). Default is (512, 512).
    """
    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        raise ValueError(f"The specified directory {directory_path} does not exist.")

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".png"):
            file_path = os.path.join(directory_path, filename)

            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Resize the image
                    resized_img = img.resize(target_size, Image.Resampling.BILINEAR)
                    # Save the resized image back to the same file
                    resized_img.save(file_path)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

resize_images_in_directory(r'F:\开题报告\Codes\Project\dataset\DatasetCHUAC\train\image')
resize_images_in_directory(r'F:\开题报告\Codes\Project\dataset\DatasetCHUAC\val\image')
