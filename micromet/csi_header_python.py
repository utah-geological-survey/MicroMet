import os

import numpy as np
from datetime import datetime, timedelta
import re

def csi_header(file):
    """
    Reads Campbell Scientific ASCII TOA5 and TOB1 headers into a dictionary.

    Args:
        file (str): Full path of TOA5 or TOB1 file.

    Returns:
        dict: A dictionary containing header information with the following keys:
            file_type, station_name, model_name, serial_number, os_version,
            dld_name, dld_signature, table_name, field_names, field_units,
            field_processing, data_types (TOB1 only), header_bytes

    Raises:
        ValueError: If there's an error in the header format.
    """
    header_bytes = 0

    # Read first line
    with open(file, 'r') as f:
        line = f.readline().strip()
        header_bytes += len(line) + 2  # add single line lengths plus CRLF
        z1 = [item.strip('"') for item in line.split(',')]

    file_type = z1[0].lower()
    if file_type == 'tob1':
        header_lines = 5
    elif file_type == 'toa5':
        header_lines = 4
    else:
        return {'file_type': 'unknown'}

    # Read next lines
    asc_head = []
    with open(file, 'r') as f:
        f.readline()  # Skip the first line
        for _ in range(header_lines - 1):
            line = f.readline().strip()
            asc_head.append(line)
            header_bytes += len(line) + 2

    # Splitting into tags
    n_fields = len(asc_head[0].split(','))
    z = [line.split(',') for line in asc_head]

    for line in z:
        if len(line) != n_fields:
            raise ValueError(f'Error in header of {file}')

    # Creating dictionary
    header = {
        'file_type': '',
        'station_name': '',
        'model_name': '',
        'serial_number': '',
        'os_version': '',
        'dld_name': '',
        'dld_signature': '',
        'table_name': '',
        'field_names': [''] * n_fields,
        'field_units': [''] * n_fields,
        'field_processing': [''] * n_fields,
        'header_bytes': header_bytes
    }

    if file_type == 'tob1':
        header['data_types'] = [''] * n_fields

    # Filling dictionary
    for i in range(8):
        header[list(header.keys())[i]] = z1[i].upper()

    for i in range(n_fields):
        header['field_names'][i] = z[0][i].strip('"').upper()
        header['field_units'][i] = z[1][i].strip('"').upper()
        header['field_processing'][i] = z[2][i].strip('"').upper()

    if file_type == 'tob1':
        for i in range(n_fields):
            header['data_types'][i] = z[3][i].strip('"').upper()

    return header


def csi_read_tob1(file, header=False, julian=False):
    """
    Reads Campbell Scientific binary TOB1 files into a Python dictionary.

    Args:
        file (str): Full path of TOB1 file.
        header (bool): If True, return TOB1 file header information.
        julian (bool): If True, add a 'julday' key to the dictionary containing the time axis as Julian date.

    Returns:
        dict: Dictionary with keys corresponding to field names from the TOB1 file.

    Raises:
        ValueError: If file doesn't exist, can't be read, or is not a TOB1 file.
    """

    if not os.path.isfile(file) or not os.access(file, os.R_OK):
        raise ValueError(f"File not existing or no read access: {file}")

    # Read file header
    header_info = csi_header(file)
    if header_info['file_type'].upper() != 'TOB1':
        raise ValueError(f"Illegal file format (TOB1 expected): {file}")

    data_block_len = os.path.getsize(file) - header_info['header_bytes']

    # Calculate size of 1 record in bytes
    n_cols = len(header_info['field_names'])
    record_bytes = sum(csi_tob_datatypes(dt, bytes=True) for dt in header_info['data_types'])

    # Calculate number of records in file
    n_rec = data_block_len // record_bytes

    # Create output dictionary
    tag_names = csi_check_tagnames(header_info['field_names'])
    data = {tag: np.zeros(n_rec, dtype=csi_tob_datatypes(dt, equivalent=True))
            for tag, dt in zip(tag_names, header_info['data_types'])}

    # Read data block in file
    with open(file, 'rb') as f:
        f.seek(header_info['header_bytes'])  # Skip header
        for r in range(n_rec):
            for c, (tag, dt) in enumerate(zip(tag_names, header_info['data_types'])):
                byte_count = csi_tob_datatypes(dt, bytes=True)
                dummy = f.read(byte_count)
                data[tag][r] = np.frombuffer(dummy, dtype=csi_tob_datatypes(dt, template=True))[0]

    # Create Julian time axis (optional)
    if julian:
        if 'seconds' in data:
            base_date = datetime(1990, 1, 1)
            if 'nanosec' in data:
                julian_array = np.array([(base_date + timedelta(seconds=s, microseconds=n/1000)).toordinal() + 1721424.5
                                         for s, n in zip(data['seconds'], data['nanosec'])])
            else:
                julian_array = np.array([(base_date + timedelta(seconds=s)).toordinal() + 1721424.5
                                         for s in data['seconds']])
            data['JULDAY'] = julian_array

    if header:
        return data, header_info
    else:
        return data





def csi_check_tagnames(tag_names):
    # Convert to uppercase and remove whitespace
    tag_names = [tag.upper().strip() for tag in tag_names]

    # Remove illegal characters
    legal_pattern = re.compile(r'[^A-Z0-9_]')
    cleaned_tags = []
    for tag in tag_names:
        cleaned = legal_pattern.sub('', tag)
        cleaned_tags.append(cleaned if cleaned else "ILLEGALCHARSONLY")

    # Make tags unique
    unique_tags = {}
    for tag in cleaned_tags:
        if tag in unique_tags:
            count = unique_tags[tag]
            unique_tags[tag] = count + 1
            yield f"{tag}_{count + 1}"
        else:
            unique_tags[tag] = 0
            yield tag



def csi_tob_datatypes(typestring, value=None, bytes=False, equivalent=False, template=False):
    """
    Look-up data type properties and conversions used in the Campbell Scientific
    TOB (Table-Oriented Binary) formats.

    Args:
        typestring (str): Valid TOB data type string.
        value (optional): Binary value read from file (format must be the same as template of the data type).

    Keywords:
        bytes (bool): If True, returns the number of bytes the data type uses in the TOB file.
        equivalent (bool): If True, returns the Python equivalent of this data type.
        template (bool): If True, returns a dummy parameter as template to read the binary data.

    Returns:
        Depends on keyword set (see above).

    Raises:
        ValueError: If an unknown data type is provided.
    """
    typestring = typestring.lower()

    if typestring in ['ieee4', 'ieee4l']:
        bytes_in_file = 4
        read_template = np.float32
        py_equivalent = float
    elif typestring in ['ulong', 'uint4']:
        bytes_in_file = 4
        read_template = np.uint32
        py_equivalent = int
    elif typestring == 'long':
        bytes_in_file = 4
        read_template = np.int32
        py_equivalent = int
    elif typestring == 'fp2':
        bytes_in_file = 2
        read_template = np.dtype(('B', 2))
        py_equivalent = float
    else:
        raise ValueError(f"Error: unknown data type in TOB file header: {typestring}")

    if bytes:
        return bytes_in_file
    elif equivalent:
        return py_equivalent
    elif template:
        return read_template
    elif value is not None:
        if typestring == 'fp2':
            return csi_fs2(value)
        else:
            return py_equivalent(value)
    else:
        return None


def csi_fs2(two_byte_array):
    # Convert two bytes to 16 bits
    bits = np.unpackbits(np.array(two_byte_array, dtype=np.uint8))

    # Extract sign, exponent, and mantissa
    s = bits[0]
    e = -1.0 * (bits[2] + 2 * bits[1])
    m = sum(bits[i] * 2 ** (15 - i) for i in range(3, 16))

    # Handle special cases
    if s == 0 and e == 0 and m == 8191:
        return float('inf')
    elif s == 1 and e == 0 and m == 8191:
        return float('-inf')
    elif s == 0 and e == 0 and m == 8190:
        return float('nan')
    elif s == 1 and e == 0 and m == 8190:
        return float('-nan')
    else:
        return (-1.0) ** s * 10.0 ** e * m

