import requests

def download_file(file_url):
    response = requests.get(file_url)
    file_path = "temp/data.csv"

    with open(file_path, "wb") as f:
        f.write(response.content)

    return file_path