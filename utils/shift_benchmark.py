import json
import gzip
import os

BENCHMARK = 5

if __name__ == "__main__":

	# Find all files in the test data directory
	files = [f for f in os.listdir("./dataset/test/") if os.path.isfile(os.path.join("./dataset/test/", f))]

	for file in files:
		with gzip.open(f"./dataset/test/{file}", "rt") as f:

			data = [json.loads(line) for line in f]

			# Recalibrate using the given benchmark
			for d in data:
				d["active"] = 1 if d["pchembl_value"] >= BENCHMARK else 0
			
			# Write the json file back in jsonl.gz format
			with gzip.open(f"./dataset/test_mod/{file}", "wt") as f:	
				for d in data:
					f.write(json.dumps(d) + "\n")