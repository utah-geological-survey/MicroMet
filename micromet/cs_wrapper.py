import subprocess
import argparse
import os

def convert_file(input_file,
                 output_file,
                 output_format,
                 fsl_file=None,
                 array_id=None,
                 format_options=None,
                 exe_path = r"C:\Program Files (x86)\Campbellsci\LoggerNet\csidft_convert.exe"
                 ):
    """
    Convert a Campbell Scientific data file using the csidft_convert.exe utility.

    This function constructs and executes the command to run csidft_convert.exe with the
    provided parameters. It handles both standard and array-based file conversions.

    https://help.campbellsci.com/loggernet-manual/ln_manual/campbell_scientific_file_formats/csidft_convert.exe.htm?

    Parameters:
    input_file (str): Path to the input file to be converted.
    output_file (str): Path where the converted file will be saved.
    output_format (str): Desired output format. Must be one of:
                         'toaci1', 'toa5', 'tob1', 'csixml', 'custom-csv', 'no-header'.
    fsl_file (str, optional): Path to the FSL file for array-based input files.
    array_id (str, optional): Array ID for array-based input files.
    format_options (int, optional): Integer value representing format options.
                                    Refer to csidft_convert.exe documentation for details.

    Returns:
    None

    Raises:
    subprocess.CalledProcessError: If the conversion process fails.

    Prints:
    Success or error messages, including any output from csidft_convert.exe.

    Example usage:
    convert_file('input.dat', 'output.csv', 'toa5', format_options=1)
    convert_file('input.dat', 'output.csv', 'toa5', fsl_file='input.fsl', array_id='1')
    """
    # Construct the base command
    command = [
        exe_path,
        input_file,
        output_file,
        output_format
    ]

    # Add optional parameters if provided
    if fsl_file:
        command.extend(["--fsl", fsl_file])
    if array_id:
        command.extend(["--array", array_id])
    if format_options is not None:
        command.extend(["--format-options", str(format_options)])

    # Execute the command
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Conversion successful.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed. Error: {e}")
        print(e.stderr)

def main():
    parser = argparse.ArgumentParser(description="Convert Campbell Scientific data files using csidft_convert.exe")
    parser.add_argument("input_file", help="Input file name")
    parser.add_argument("output_file", help="Output file name")
    parser.add_argument("output_format", choices=["toaci1", "toa5", "tob1", "csixml", "custom-csv", "no-header"],
                        help="Output format")
    parser.add_argument("--fsl", help="FSL file for array-based input files")
    parser.add_argument("--array", help="Array ID for array-based input files")
    parser.add_argument("--format-options", type=int, help="Format options (integer value)")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return

    convert_file(args.input_file, args.output_file, args.output_format,
                 args.fsl, args.array, args.format_options)

if __name__ == "__main__":
    main()