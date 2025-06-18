import re
import json

def remove_unusual_terminators(text):
    # Remove Unicode Line Separator (\u2028), Paragraph Separator (\u2029)
    return re.sub(r'[\u2028\u2029]', '', text)

input_file = 'wildchat_prompt_response_pairs.jsonl'
output_file = 'wildchat_prompt_response_pairs_clean.jsonl'

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        obj = json.loads(line)
        for key, val in obj.items():
            if isinstance(val, str):
                obj[key] = remove_unusual_terminators(val)
        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f"Cleaned file saved to {output_file}")

