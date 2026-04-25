import subprocess, sys

result = subprocess.run(
    [sys.executable, 'wrapper.py', 'install'],
    capture_output=True, text=True
)

print("Now go to Runtime and Restart Runtime")
