import os
import re
import unicodedata
import sys

def preprocess_text(text):
    # Remove BOM
    text = text.replace('\ufeff', '')

    # Remove Gutenberg header and footer
    header_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    footer_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
    start_idx = text.find(header_marker)
    end_idx = text.rfind(footer_marker)
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx + len(header_marker):end_idx]

    # Unicode normalization
    text = unicodedata.normalize('NFKD', text)

    # Optional: Remove specific problematic characters or patterns
    # text = text.replace('â', "'")  # Example: replacing a specific problematic pattern

    # Standardize line breaks and remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_books(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, 'r', encoding='utf-8-sig') as file:  # Note the 'utf-8-sig' encoding
                book_text = file.read()

            processed_text = preprocess_text(book_text)

            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(processed_text)
            print(f"Processed and saved: {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_script.py <input_folder> <output_folder>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    preprocess_books(input_directory, output_directory)
