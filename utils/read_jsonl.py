import json
import gzip
import argparse

# Read jsonl.gz file into a list of dictionaries and save it.
def read_jsonl_gz(file_path: str):
	with gzip.open(file_path, "rt") as f:
		# Save the jsonl.gz. into a readable json file
		data = [json.loads(line) for line in f]
		print(f"{data}")


# Take jsonl.gz. file as input and save it as a json file
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# Take command line argument for file_path
	parser.add_argument("--chembl_id", type=str)
	args = parser.parse_args()
	read_jsonl_gz(f"./dataset/test/{args.chembl_id}.jsonl.gz")