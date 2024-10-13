import os
import cv2
from inference_sdk import InferenceHTTPClient

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="ZGTSq44QFxJo3d6zVITd"
)

# Path to the single image you want to process
image_path = 'D:\\Intern_Image\\Healthy Potatoes\\29.jpg'  # Change this to your image
output_folder = 'D:\\Intern_Image'  # Folder to save output image

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Run inference on the single image using the API
result = CLIENT.infer(image_path, model_id="potato-seg/1")

# Count the number of detected potatoes
potato_count = len([pred for pred in result['predictions'] if pred['class'] == 'potato'])

# Print the result
image_file = os.path.basename(image_path)  # Get the image filename
print(f"{image_file}: Detected potatoes: {potato_count}")

# Load the image using OpenCV
img = cv2.imread(image_path)

# Optionally display the object count on the image at the bottom-right corner
text = f'Count: {potato_count}'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

# Get the text size (width, height) to position it at the bottom-right corner
text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
text_x = img.shape[1] - text_size[0] - 10  # 10 pixels padding from the right
text_y = img.shape[0] - 10  # 10 pixels padding from the bottom

# Put the count text on the image
cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

# Save the image with the object count displayed
output_image_path = os.path.join(output_folder, image_file.split('.')[0] + '_counted.jpg')
cv2.imwrite(output_image_path, img)

print(f'Saved processed image with object count: {output_image_path}')

# ----- Code for Single Image ----- #




# import os
# import cv2
# from inference_sdk import InferenceHTTPClient

# # Initialize the client
# CLIENT = InferenceHTTPClient(
#     api_url="https://outline.roboflow.com",
#     api_key="ZGTSq44QFxJo3d6zVITd"
# )

# # Folder containing multiple images
# image_folder = 'D:\\Intern_Image\\Healthy Potatoes'
# output_folder = 'D:\\Intern_Image\\roboflowimgoutput'  # Folder to save output images

# # Ensure the output folder exists
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Loop through all images in the folder
# for image_file in os.listdir(image_folder):
#     # Check if the file is an image
#     if image_file.endswith('.jpg') or image_file.endswith('.png'):
#         # Full path to the image
#         image_path = os.path.join(image_folder, image_file)

#         # Run inference on the image using the API
#         result = CLIENT.infer(image_path, model_id="potato-seg/1")

#         # Count the number of detected potatoes
#         potato_count = len([pred for pred in result['predictions'] if pred['class'] == 'potato'])

#         # Print the result
#         print(f"{image_file}: Detected potatoes: {potato_count}")

#         # Load the image using OpenCV
#         img = cv2.imread(image_path)

#         # Optionally display the object count on the image at the bottom-right corner
#         text = f'Count: {potato_count}'
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 1
#         font_thickness = 2

#         # Get the text size (width, height) to position it at the bottom-right corner
#         text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
#         text_x = img.shape[1] - text_size[0] - 10  # 10 pixels padding from the right
#         text_y = img.shape[0] - 10  # 10 pixels padding from the bottom

#         # Put the count text on the image
#         cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

#         # Save the image with the object count displayed
#         output_image_path = os.path.join(output_folder, image_file.split('.')[0] + '_counted.jpg')
#         cv2.imwrite(output_image_path, img)

#         print(f'Saved processed image with object count: {output_image_path}')
