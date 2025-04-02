import os
import csv

# Automatically get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set paths relative to the script location
text_directory = os.path.join(script_dir, 'text')  # Folder containing text files
output_directory = os.path.join(script_dir, 'data512')  # Output folder for CSV files

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Process all .txt files in the text directory
for filename in os.listdir(text_directory):
    if filename.endswith('.txt'):
        # Read the content of the text file
        with open(os.path.join(text_directory, filename), 'r', encoding='utf-8') as f:
            content = f.read()

        # Split text into chunks of 512 characters
        chunk_size = 512
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

        # Add remaining content if not evenly divisible by chunk size
        if len(content) % chunk_size != 0:
            chunks.append(content[-(len(content) % chunk_size):])

        # Create corresponding CSV file and write content
        csv_filename = os.path.splitext(filename)[0] + '.csv'
        with open(os.path.join(output_directory, csv_filename), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['content'])  # Write header
            for chunk in chunks[:-1]:  # Write all chunks except the last one
                writer.writerow([chunk])

        print(f'{csv_filename} saved successfully')