import os


def get_file_paths(directory):
    file_paths = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_paths.append(os.path.join(directory, filename))
    return file_paths