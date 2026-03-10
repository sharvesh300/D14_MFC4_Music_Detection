import redis

r = redis.Redis()

total = 0
count = 0
max_len = 0

for key in r.scan_iter("fp:*"):
    length = r.llen(key)
    total += length
    count += 1
    max_len = max(max_len, length)

print("keys:", count)
print("total fingerprints:", total)
print("avg bucket size:", total / count)
print("largest bucket:", max_len)
