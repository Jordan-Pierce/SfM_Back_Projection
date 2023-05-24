import subprocess

# Define the command to install the requirements
commands = ['C:\\Program Files\\Agisoft\\Metashape Pro\\python\\python.exe',
            '-m',
            'pip',
            'install',
            '-r',
            'requirements.txt']

# Use subprocess to run the command in the terminal
result = subprocess.run(commands, shell=True, capture_output=True)

# Check if the command was successful and output the result
if result.returncode == 0:
    print("Requirements installed successfully!")
else:
    print("Failed to install requirements. Error message:")
    print(result.stdout.decode() if result.stdout else result.stderr.decode())
