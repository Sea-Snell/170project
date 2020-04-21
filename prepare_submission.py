import sys
import os
import json
from parse import validate_file

if __name__ == '__main__':
    outputs_dir = sys.argv[1]
    submission_name = sys.argv[2]
    submission = {}
    for input_path in os.listdir("inputs"):
        graph_name = input_path.split('.')[0]
        output_file = f'{outputs_dir}/{graph_name}.out'
        if os.path.exists(output_file) and validate_file(output_file):
            output = open(f'{outputs_dir}/{graph_name}.out').read()
            submission[input_path] = output
    with open(submission_name, 'w') as f:
        f.write(json.dumps(submission))
