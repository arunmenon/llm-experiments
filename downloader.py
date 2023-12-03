import requests

# Function to download a book from Project Gutenberg
def download_gutenberg_text(book_id):
    url = f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error downloading {book_id}: {e}")
        return None

# Read the list of books from the file
def get_book_ids(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [line.split(',')[0].strip() for line in lines]

# Main function to download books
def download_books(file_path):
    book_ids = get_book_ids(file_path)
    for book_id in book_ids:
        book_text = download_gutenberg_text(book_id)
        if book_text:
            file_name = f'book_{book_id}.txt'
            with open(file_name, 'w') as file:
                file.write(book_text)
            print(f"Downloaded and saved: {file_name}")

# Call the download_books function with your file path
download_books('books.txt')
