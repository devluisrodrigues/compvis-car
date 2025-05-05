import os

# Directory containing the images
directory = '/home/joao/Documents/VisComp/compvis-car/imgs'

# Get a sorted list of files in the directory
files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])

# Rename files to 1-n.png
for i, filename in enumerate(files, start=1):
    old_path = os.path.join(directory, filename)
    new_filename = f"{i}.png"
    new_path = os.path.join(directory, new_filename)
    os.rename(old_path, new_path)

print("Files renamed successfully.")