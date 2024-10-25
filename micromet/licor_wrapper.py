
import argparse
import subprocess
import os
import sys


def run_eddypro(system="win", mode="desktop", caller="console", environment=None, proj_file=None):
    """
    Run the EddyPro engine with specified parameters.

    Args:
    system (str): Operating system. Options: 'win', 'linux', 'mac'. Default is 'win'.
    mode (str): Running mode. Options: 'embedded', 'desktop'. Default is 'desktop'.
    caller (str): Caller type. Options: 'gui', 'console'. Default is 'console'.
    environment (str): Working directory for embedded mode. Default is None.
    proj_file (str): Path to the project file (*.eddypro). Default is None.

    Returns:
    subprocess.CompletedProcess: Result of the subprocess run.
    """
    # Construct the command
    command = ["eddypro_rp"]

    if system != "win":
        command.extend(["-s", system])

    if mode != "desktop":
        command.extend(["-m", mode])

    if caller != "console":
        command.extend(["-c", caller])

    if environment:
        command.extend(["-e", environment])

    if proj_file:
        command.append(proj_file)

    # Run the command
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("EddyPro executed successfully.")
        print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error executing EddyPro: {e}")
        print(e.stderr)
        return e


def main():
    parser = argparse.ArgumentParser(description="Run EddyPro engine from command line")
    parser.add_argument("-s", "--system", choices=["win", "linux", "mac"], default="win",
                        help="Operating system (default: win)")
    parser.add_argument("-m", "--mode", choices=["embedded", "desktop"], default="desktop",
                        help="Running mode (default: desktop)")
    parser.add_argument("-c", "--caller", choices=["gui", "console"], default="console",
                        help="Caller type (default: console)")
    parser.add_argument("-e", "--environment", help="Working directory for embedded mode")
    parser.add_argument("proj_file", nargs="?", help="Path to project file (*.eddypro)")

    args = parser.parse_args()

    # Ensure eddypro_rp is in the system PATH
    if not any(os.access(os.path.join(path, "eddypro_rp"), os.X_OK) for path in os.environ["PATH"].split(os.pathsep)):
        print(
            "Error: eddypro_rp is not found in the system PATH. Please add the EddyPro binary directory to your PATH.")
        sys.exit(1)

    run_eddypro(args.system, args.mode, args.caller, args.environment, args.proj_file)


if __name__ == "__main__":
    main()