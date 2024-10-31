import os
import sys


def process_files(directory, original_class_num):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()

            lines_to_delete = []
            numbers_dict = {}

            # Identify lines to delete and build a dictionary for matching numbers
            for line in lines:
                parts = line.split()
                if len(parts) > 1:
                    first_num = int(parts[0])
                    rest_of_line = ' '.join(parts[1:])
                    if first_num > original_class_num:
                        lines_to_delete.append(line)
                    #else:
                    if rest_of_line in numbers_dict:
                        numbers_dict[rest_of_line].append(first_num)
                    else:
                        numbers_dict[rest_of_line] = [first_num]

            # Process deletions and modifications
            modified_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) > 1:
                    rest_of_line = ' '.join(parts[1:])
                    if line not in lines_to_delete:
                        if rest_of_line in numbers_dict and len(numbers_dict[rest_of_line]) > 1:
                            modified_lines.append(f"{line.strip()} {' '.join(map(str, numbers_dict[rest_of_line][1:]))}")
                            numbers_dict[rest_of_line] = [numbers_dict[rest_of_line][0]]  # Keep only the first element
                        else:
                            modified_lines.append(line.strip())
            
            # Write the modified lines back to the file
            with open(filepath, 'w') as file:
                for line in modified_lines:
                    file.write(line + '\n')

if __name__ == '__main__':
    directory = sys.argv[1]
    original_class_num = int(sys.argv[2])
    process_files(directory, original_class_num-1)

