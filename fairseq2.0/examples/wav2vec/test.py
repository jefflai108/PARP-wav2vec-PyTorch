
with open("data/commonvoice-all-train/all-train.tsv", "r") as f:
    f.readline()
    content = f.readlines()
content = [x.strip('\n') for x in content]

best = 0
threshold = 100000
count = 0
for i in content:
    num_frames = int(i.split()[1])
    if num_frames >= best:
        best = num_frames
    if num_frames >= threshold:
        print(num_frames)
        count += 1
print(best)
print(count, len(content))
