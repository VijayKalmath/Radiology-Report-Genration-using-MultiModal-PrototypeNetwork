import json
import re
from collections import Counter

ann = json.loads(open('Data/iu_xray/annotation.json', 'r').read())

for ele in ann['train']:
    print(ele['report'])

    