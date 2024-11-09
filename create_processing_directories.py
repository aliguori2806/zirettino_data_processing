import os
import sys

def create_processing_directories(working_dir_path):
    # Create the main directory if it doesn't exist
    os.makedirs(working_dir_path, exist_ok=True)
    
    # List of subdirectories to create
    subdirectories = [
        "data_processing_scripts",
        "logs",
        "gains",
        "pedestals",
        "rootfiles",
        "analysis",
        "asic_array_info"
    ]
    
    # Create each subdirectory inside the working directory
    for subdir in subdirectories:
        os.makedirs(os.path.join(working_dir_path, subdir), exist_ok=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <working_dir_path>")
    else:
        working_dir_path = sys.argv[1]
        create_processing_directories(working_dir_path)
