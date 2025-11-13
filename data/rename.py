import os

# Set the directory where the folders are located
directory = "audio"  # Change this to your actual directory path

# Iterate through all items in the directory
for folder_name in os.listdir(directory):
    folder_path = os.path.join(directory, folder_name)

    # Check if it is a directory and contains '_sound'
    if os.path.isdir(folder_path) and "-" in folder_name:
        new_folder_name = folder_name.replace("-", " ",)
        new_folder_path = os.path.join(directory, new_folder_name)

        # Rename the folder
        os.rename(folder_path, new_folder_path)
        print(f'Renamed: "{folder_name}" -> "{new_folder_name}"')

print("Renaming completed!")
