
max_weight = 5000
curr_weight = 0
ret = 0

i = 777
while i > 0:
    curr_weight += i
    if curr_weight >= max_weight:
        i -= 1
        ret += 1
        curr_weight = 0
        continue
    i -= 1
print(ret)