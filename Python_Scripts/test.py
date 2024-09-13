#%%
import os

# Get the current directory's path
current_directory = "/home/ubuntu/WildFire_Capstone/Output/Fire_Perimeter"

# List everything in the current directory
entries = os.listdir(current_directory)

# Filter the list to include only files (excluding directories)
files = [entry for entry in entries if os.path.isfile(os.path.join(current_directory, entry))]

# Count the number of files
number_of_files = len(files)

print(f"Number of files in the current directory: {number_of_files}")

# %%
