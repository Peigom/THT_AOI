import json

# Load the JSON file
def pretty_json(path):
    file_path = path
    with open(file_path, "r") as f:
        data = json.load(f)

    # Pretty print the JSON content
    pretty_json = json.dumps(data, indent=4)

    # Save the prettified JSON to a new file
    pretty_file_path = path
    with open(pretty_file_path, "w") as f:
        f.write(pretty_json)

