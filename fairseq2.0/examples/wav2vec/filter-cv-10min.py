import sys
import random

with open(sys.argv[1], 'r') as f:
    head = f.readline().strip('\n')
    tsv_content = f.readlines()
tsv_content = [x.strip('\n') for x in tsv_content]

with open(sys.argv[2], 'r') as f:
    ltr_content = f.readlines()
ltr_content = [x.strip('\n') for x in ltr_content]

# Shuffle two lists with same order
# Using zip() + * operator + shuffle()
temp = list(zip(tsv_content, ltr_content))
random.shuffle(temp)
r_tsv_content, r_ltr_content = zip(*temp)

r_tsv_content = r_tsv_content[:len(r_tsv_content)//6+1]
r_ltr_content = r_ltr_content[:len(r_ltr_content)//6+1]


with open(sys.argv[3], 'w') as f:
    f.write('%s\n' % head)
    for x in r_tsv_content:
        f.write('%s\n' % x)

with open(sys.argv[4], 'w') as f:
    f.write('%s\n' % head)
    for x in r_ltr_content:
        f.write('%s\n' % x)
