import os
import json

UNFILTERED_DIR = "/Users/andrew/Code/ersilia/bioassays/dataset/test_threshold6_unfiltered"
FILTERED_FILENAMES = "/Users/andrew/Code/ersilia/bioassays/dataset/test_filtered_thresh6.json"
OUT_DIR = "/Users/andrew/Code/ersilia/bioassays/dataset/test_threshold6_filtered"

if __name__ == "__main__":

	# Find all files in the test data directory
	files = [f for f in os.listdir(UNFILTERED_DIR) if os.path.isfile(os.path.join(UNFILTERED_DIR, f))]

	# If filename is in filtered filenames, add it to the output directory
	os.makedirs(OUT_DIR, exist_ok=True)
	filtered_filenames = set(json.load(open(FILTERED_FILENAMES)))
	for f in files:
		if f in filtered_filenames:
			# Copy the file to the output directory using json
			print(f"Copying {f} to {OUT_DIR}.")
			os.system(f"cp {UNFILTERED_DIR}/{f} {OUT_DIR}/{f}")