import re, sys
p = sys.argv[1]
s = open(p).read()
if "patched: output was never defined" in s:
    print("already patched"); sys.exit(0)
a = "    if 'SAMPLER' in os.environ:\n"
b = "        fname = os.path.join(base_dir, 'convergence.txt')\n"
assert a in s and b in s, "UltraNest.py differs from what the patch expects"
s = s.replace(a, "    output = None  # patched: output was never defined (the nested_run() call that set it is commented out)\n" + a, 1)
s = s.replace(b, "        os.makedirs(base_dir, exist_ok=True)\n" + b, 1)
open(p, "w").write(s); print("patched", p)
