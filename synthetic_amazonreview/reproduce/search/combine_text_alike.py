

import argparse
import os

def combine_jsonl(input_files, output_file):
    file_handles = [open(file, 'r') for file in input_files]

    with open(output_file, 'w') as outfile:
        while True:
            lines_written = False
            for f in file_handles:
                line = f.readline()
                if line:
                    outfile.write(line)
                    lines_written = True
            if not lines_written:
                break

    for f in file_handles:
        f.close()

def main():
    parser = argparse.ArgumentParser(description="Combine multiple JSONL files in a round-robin fashion.")
    parser.add_argument('--input_files', nargs='+', help="List of input JSONL files to combine")
    parser.add_argument('--output_file', help="Output JSONL file")
    args = parser.parse_args()

    combine_jsonl(args.input_files, args.output_file)

if __name__ == "__main__":
    main()

