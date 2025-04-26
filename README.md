# NEURAL-STYLE-TRANSFER

"Company":CODTECH IT SOLUTIONS

"NAME":MOHAMMED NASHRA SANIYA

"INTERN ID":CODF167

"DOMAIN":ARTIFICIAL INTELLIGENCE MARKUP LANGUAGE

"DURATION:"4 WEEKS

"MENTOR":NEELA SANTHOSH

# Project Description

Overview

This project implements Neural Style Transfer (NST), an innovative technique that leverages deep learning to combine the content of one image with the style of another. The goal is to create an output image that retains the content of a "content" image while adopting the artistic style of a "style" image. This approach uses pre-trained Convolutional Neural Networks (CNNs) like VGG19 to extract features from the images and perform optimization to generate a new image that merges both content and style.

The Neural Style Transfer method essentially works by minimizing the difference between:

1. The content loss (the difference between the content of the output and the content image).

2. The style loss (the difference between the style of the output and the style image).

The model then iteratively refines an initially random image to converge on a visually pleasing result.

# Requirements

1. Python 3.x

2. PyTorch

3. torchvision

4. PIL (Pillow)

5. Matplotlib

You can install the required dependencies by running the following:

-->  pip install torch torchvision pillow matplotlib

# How It Works

1. Load Images: A content image (e.g., a photograph) and a style image (e.g., a painting) are loaded and preprocessed (resized and normalized).

2. Feature Extraction: The pre-trained VGG19 model is used to extract features from multiple layers of the images. The layers are chosen to capture both low-level (textures, patterns) and high-level (shapes, structures) features.

3. Loss Calculation: Two types of losses are computed:

4. Content Loss: Measures the difference in content between the content image and the generated image.

5. Style Loss: Measures the difference in style between the style image and the generated image using Gram matrices.

6. Optimization: The generated image is optimized by minimizing a weighted sum of both content and style losses.

7. Result: After several iterations, the generated image is refined to combine the content of the original image with the style of the chosen artwork.

# Key Features

1. Content Preservation: The final image retains the structure and content of the content image.

2. Artistic Style: The generated image incorporates the visual characteristics (such as colors, patterns, textures) of the style image.

3. Deep Learning Framework: Utilizes PyTorch for the implementation, leveraging the power of CNNs and GPU acceleration.

4. Pre-trained VGG19 Model: Uses VGG19 as a feature extractor for both content and style images.

# How to Run the Project

1. Download the content and style images: Place the images you want to use in the project folder.

2. Configure paths: Set the file paths for the content and style images in the code.

3. Run the script: Execute the script using the command:

      python neural_style_transfer.py

4. The output image will be saved in the output/ folder, and you can view the result using matplotlib or any image viewer.

# Example Usage

Given a content image content.jpg and a style image style.jpg, the algorithm combines them into a new image with the content of content.jpg and the style of style.jpg.

Example:

python neural_style_transfer.py

# Conclusion

Neural Style Transfer is a powerful and creative application of deep learning that allows you to transform regular images into artistic representations. This project demonstrates the use of CNNs in generating art by merging content and style using optimization techniques.

# Credits

1. VGG19 model pre-trained on ImageNet used for feature extraction.

2. PyTorch: Framework for implementing the model and optimization.

3. Original paper: Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style.

# output
