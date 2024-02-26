"""
Rename photos with polish characters to remove them
"""
import os
import shutil
from config_variables import data_path

if __name__ == "__main__":
    # Get a list of all jpg files in the data directory
    data_dir = f'../{data_path}'
    data = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if ".jpg" in file:
                path = os.path.join(root, file)
                data.append({"file": path})

    for file in data:
        new_file = file['file'].replace('ą', 'a').replace('ć', 'c').replace('ę', 'e').replace('ł', 'l').replace('ń', 'n').replace('ó', 'o').replace('ś', 's').replace('ź', 'z').replace('ż', 'z')
        shutil.move(file['file'], new_file)
        print(f"Renamed {file['file']} to {new_file}")