import reprlib
from timeit import timeit

big_list = list(range(10000))
big_dict = {f"k{i}": f"v{i}" for i in range(1000)}
big_str = "x" * 10000

def old_approach(value):
    s = str(value)
    return s if len(s) < 500 else s[:500] + "..."

def new_approach(value):
    if isinstance(value, str):
        return value[:200] + "..." if len(value) > 200 else value
    return reprlib.repr(value)

print("=== str() + manual truncation vs reprlib.repr ===\n")

for label, val in [("big_list (10k items)", big_list), ("big_dict (1k items)", big_dict), ("big_str (10k chars)", big_str)]:
    old_time = timeit(lambda: old_approach(val), number=10000)
    new_time = timeit(lambda: new_approach(val), number=10000)
    speedup = old_time / new_time
    print(f"{label}:")
    print(f"  old: {old_time*1000:.3f}ms, new: {new_time*1000:.3f}ms ({speedup:.1f}x)")
    print(f"  old output: {old_approach(val)[:60]}...")
    print(f"  new output: {new_approach(val)[:60]}...")
    print()
