import os
import glob
import pandas as pd

def get_nth_latest_file(folder_path, file_extension='*.csv', n=2):
    """Get the nth latest file from a folder."""
    list_of_files = glob.glob(os.path.join(folder_path, file_extension))
    sorted_files = sorted(list_of_files, key=os.path.getctime, reverse=True)
    if len(sorted_files) >= n:
        return sorted_files[n - 1]  # Get the nth latest file (n=2 means second latest)
    else:
        raise FileNotFoundError(f"Less than {n} files found in the folder: {folder_path}")

def load_data(folder_path, n=1):
    """Load the nth latest loan dataset (default is the latest dataset)."""
    file_path = get_nth_latest_file(folder_path, n=n)
    print(f"Loading data from: {file_path}")
    loan_data = pd.read_csv(file_path)
    return loan_data

# Example usage:
# folder_path = 'C:\\path\\to\\data\\folder'
# loan_data = load_data(folder_path)