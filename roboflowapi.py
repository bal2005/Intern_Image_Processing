from inference_sdk import InferenceHTTPClient

from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="#########"
)

result = CLIENT.infer( "D:\\Intern_Image\\Healthy Potatoes\\29.jpg",model_id="potato-detection-3et6q/11")
print(result)
# Count the number of detected potatoes
potato_count = len([pred for pred in result['predictions'] if pred['class'] == 'Potato'])

print(f"Detected potatoes: {potato_count}")

# ----- Code for Single Image ----- #

# import os
# import csv
# from inference_sdk import InferenceHTTPClient

# # Initialize the client
# CLIENT = InferenceHTTPClient(
#     api_url="https://outline.roboflow.com",
#     api_key="ZGTSq44QFxJo3d6zVITd"
# )

# # Folder containing multiple images
# image_folder = 'D:\\Intern_Kurtosh\\Healthy Potatoes'
# output_csv = 'D:\\Intern_Kurtosh\\potato_count.csv'  # Path to save the CSV file

# # Open a CSV file to write
# with open(output_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)
    
#     # Write the header row
#     writer.writerow(["Image Name", "Potato Count"])

#     # Loop through all images in the folder
#     for image_file in os.listdir(image_folder):
#         # Check if the file is an image
#         if image_file.endswith('.jpg') or image_file.endswith('.png'):
#             # Full path to the image
#             image_path = os.path.join(image_folder, image_file)

#             # Run inference on the image
#             result = CLIENT.infer(image_path, model_id="potato-seg/1")

#             # Count the number of detected potatoes
#             potato_count = len([pred for pred in result['predictions'] if pred['class'] == 'potato'])

#             # Print the result
#             print(f"{image_file}: Detected potatoes: {potato_count}")

#             # Write the result to the CSV file
#             writer.writerow([image_file, potato_count])

# print(f"Potato count for all images saved in: {output_csv}")
