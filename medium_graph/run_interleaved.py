#!/usr/bin/env python3
"""Run commands from a light-to-heavy script using a deque: x threads pull
from the light end, y threads pull from the heavy end, to saturate the GPU
without OOMing.

Usage: ./run_interleaved.py script.sh [-j 2:1] [-- args...]
"""
import argparse, os, subprocess, sys
from collections import deque
from threading import Thread

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("script")
ap.add_argument("-j", default="2:1",
                help="light:heavy concurrent jobs (default: 2:1)")
ap.add_argument("script_args", nargs="*",
                help="positional args passed to the script ($1, $2, ...)")
args = ap.parse_args()

x, y = (int(n) for n in args.j.split(":"))

# parse commands; honour export lines as env vars
commands = deque()
with open(args.script) as f:
    for line in f:
        s = line.strip()
        # ignore comments
        if s.startswith("#"):
            pass
        # handle environment variables
        elif s.startswith("export "):
            k, _, v = s[7:].partition("=")
            os.environ[k] = v
        # if not a blank line
        elif s:
            # substitute $1, $2, ... with positional args
            for i, arg in enumerate(args.script_args, 1):
                s = s.replace(f"${i}", arg)
            # use the current interpreter for "python" commands
            if s.startswith("python "):
                s = sys.executable + s[6:]
            commands.append(s)

def worker(label):
    while True:
        try:
            if label == "light":
                cmd = commands.popleft()
            elif label == "heavy":
                cmd = commands.pop()
        except IndexError:
            break
        print(f"[{label}] {cmd}")
        subprocess.run(cmd, shell=True)

threads = ([Thread(target=worker, args=("light",)) for _ in range(x)]
         + [Thread(target=worker, args=("heavy",)) for _ in range(y)])
for t in threads:
    t.start()
for t in threads:
    t.join()
