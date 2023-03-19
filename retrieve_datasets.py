import requests
import json
import zipfile
import os


def download_data(url, file_path):
    """Download data from URL and save to the specified file path."""
    print(f"Downloading {url}")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download data from {url}")
    with open(file_path, "wb") as f:
        f.write(response.content)


def parse_data(file_path):
    """Parse data from the specified file path and return as a list of rows."""
    with open(file_path, "r") as f:
        data = f.read()
    # Each field is separated by a space
    # Each row is separated by a new line
    data = data.split("\n")
    # Remove spaces before and after each field
    data = [d.strip() for d in data]
    data = [d.split(" ") for d in data]
    # Remove '' from values in each row
    data = [[v for v in row if v != ""] for row in data]
    return data


def save_data_as_csv(file_path, data):
    """Save the specified data to a CSV file at the specified file path."""
    with open(file_path, "w") as f:
        for row in data:
            f.write(",".join(row) + "\n")


def retrieve_datasets():
    """Retrieve the datasets and save them to CSV files."""
    types = {
        'S': ['s1', 's2', 's3', 's4'],
        'A': ['a1', 'a2', 'a3'],
        'G2': ['g2-1-10.txt', 'g2-1-100.txt', 'g2-1-20.txt', 'g2-1-30.txt', 'g2-1-40.txt', 'g2-1-50.txt', 'g2-1-60.txt',
               'g2-1-70.txt', 'g2-1-80.txt', 'g2-1-90.txt', 'g2-1024-10.txt', 'g2-1024-100.txt', 'g2-1024-20.txt',
               'g2-1024-30.txt', 'g2-1024-40.txt', 'g2-1024-50.txt', 'g2-1024-60.txt', 'g2-1024-70.txt',
               'g2-1024-80.txt', 'g2-1024-90.txt', 'g2-128-10.txt', 'g2-128-100.txt', 'g2-128-20.txt', 'g2-128-30.txt',
               'g2-128-40.txt', 'g2-128-50.txt', 'g2-128-60.txt', 'g2-128-70.txt', 'g2-128-80.txt', 'g2-128-90.txt',
               'g2-16-10.txt', 'g2-16-100.txt', 'g2-16-20.txt', 'g2-16-30.txt', 'g2-16-40.txt', 'g2-16-50.txt',
               'g2-16-60.txt', 'g2-16-70.txt', 'g2-16-80.txt', 'g2-16-90.txt', 'g2-2-10.txt', 'g2-2-100.txt',
               'g2-2-20.txt', 'g2-2-30.txt', 'g2-2-40.txt', 'g2-2-50.txt', 'g2-2-60.txt', 'g2-2-70.txt', 'g2-2-80.txt',
               'g2-2-90.txt', 'g2-256-10.txt', 'g2-256-100.txt', 'g2-256-20.txt', 'g2-256-30.txt', 'g2-256-40.txt',
               'g2-256-50.txt', 'g2-256-60.txt', 'g2-256-70.txt', 'g2-256-80.txt', 'g2-256-90.txt', 'g2-32-10.txt',
               'g2-32-100.txt', 'g2-32-20.txt', 'g2-32-30.txt', 'g2-32-40.txt', 'g2-32-50.txt', 'g2-32-60.txt',
               'g2-32-70.txt', 'g2-32-80.txt', 'g2-32-90.txt', 'g2-4-10.txt', 'g2-4-100.txt', 'g2-4-20.txt',
               'g2-4-30.txt', 'g2-4-40.txt', 'g2-4-50.txt', 'g2-4-60.txt', 'g2-4-70.txt', 'g2-4-80.txt', 'g2-4-90.txt',
               'g2-512-10.txt', 'g2-512-100.txt', 'g2-512-20.txt', 'g2-512-30.txt', 'g2-512-40.txt', 'g2-512-50.txt',
               'g2-512-60.txt', 'g2-512-70.txt', 'g2-512-80.txt', 'g2-512-90.txt', 'g2-64-10.txt', 'g2-64-100.txt',
               'g2-64-20.txt', 'g2-64-30.txt', 'g2-64-40.txt', 'g2-64-50.txt', 'g2-64-60.txt', 'g2-64-70.txt',
               'g2-64-80.txt', 'g2-64-90.txt', 'g2-8-10.txt', 'g2-8-100.txt', 'g2-8-20.txt', 'g2-8-30.txt',
               'g2-8-40.txt', 'g2-8-50.txt', 'g2-8-60.txt', 'g2-8-70.txt', 'g2-8-80.txt', 'g2-8-90.txt'],
        'Birtch': ['birch1', 'birch2', 'birch3'],
        'DIM': ['dim032', 'dim064', 'dim128', 'dim256', 'dim512', 'dim1024'],
        'Unbalance': ['unbalance']
    }

    root_url = "http://cs.uef.fi/sipu/datasets/"

    for t, files in types.items():
        if t == "G2":
            zip_url = "http://cs.uef.fi/sipu/datasets/g2-txt.zip"
            zip_file_path = "datasets/g2-gt-txt.zip"
            download_data(zip_url, zip_file_path)
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall("datasets/")
            for file_name in files:
                file_path = f"datasets/{file_name}"
                data = parse_data(file_path)
                csv_file_path = f"datasets/{file_name.split('.')[0]}.csv"
                save_data_as_csv(csv_file_path, data)
                types[t][types[t].index(file_name)] = csv_file_path
                os.remove(file_path)
        else:
            for file_name in files:
                url = f"{root_url}{file_name}.txt"
                file_path = f"datasets/{file_name}.txt"
                download_data(url, file_path)
                data = parse_data(file_path)
                csv_file_path = f"datasets/{file_name}.csv"
                save_data_as_csv(csv_file_path, data)
                types[t][types[t].index(file_name)] = csv_file_path
                os.remove(file_path)

    with open("datasets/types.json", "w") as f:
        json.dump(types, f)


if __name__ == "__main__":
    retrieve_datasets()
