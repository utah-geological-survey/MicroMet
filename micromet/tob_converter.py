import os
import numpy as np
from datetime import datetime, timedelta
import re

import struct

import datetime as datetime
from operator import itemgetter

__author__ = 'spirro00'

def csi_header(file):
    """
    Read the header of a Campbell Scientific TOA5 or TOB1 file.

    Args:
        file (str): Path to the TOA5 or TOB1 file.

    Returns:
        dict: A dictionary containing header information with keys:
            file_type, station_name, model_name, serial_number, os_version,
            dld_name, dld_signature, table_name, field_names, field_units,
            field_processing, data_types (TOB1 only), header_bytes

    Raises:
        ValueError: If the file format is neither TOA5 nor TOB1.
    """
    with open(file, 'r') as f:
        lines = [f.readline().strip() for _ in range(5)]

    header_bytes = sum(len(line) + 2 for line in lines)
    z1 = [item.strip('"') for item in lines[0].split(',')]
    file_type = z1[0].lower()

    if file_type not in ['tob1', 'toa5']:
        return {'file_type': 'unknown'}

    header_lines = 5 if file_type == 'tob1' else 4
    z = [line.split(',') for line in lines[1:header_lines]]
    n_fields = len(z[0])

    header = {
        'file_type': file_type,
        'station_name': z1[1],
        'model_name': z1[2],
        'serial_number': z1[3],
        'os_version': z1[4],
        'dld_name': z1[5],
        'dld_signature': z1[6],
        'table_name': z1[7],
        'field_names': [item.strip('"').upper() for item in z[0]],
        'field_units': [item.strip('"').upper() for item in z[1]],
        'field_processing': [item.strip('"').upper() for item in z[2]],
        'header_bytes': header_bytes
    }

    if file_type == 'tob1':
        header['data_types'] = [item.strip('"').upper() for item in z[3]]

    return header


def csi_read_tob1(file, header=False, julian=False):
    """
    Read a Campbell Scientific TOB1 (Table-Oriented Binary) file.

    Args:
        file (str): Path to the TOB1 file.
        header (bool, optional): If True, return the file header information. Defaults to False.
        julian (bool, optional): If True, add a 'JULDAY' key to the data dictionary
                                 containing the time axis as Julian date. Defaults to False.

    Returns:
        dict or tuple: If header is False, returns a dictionary of data arrays.
                       If header is True, returns a tuple (data_dict, header_dict).

    Raises:
        ValueError: If the file is not a TOB1 file.
    """
    header_info = csi_header(file)
    if header_info['file_type'].upper() != 'TOB1':
        raise ValueError(f"Illegal file format (TOB1 expected): {file}")

    data_block_len = os.path.getsize(file) - header_info['header_bytes']
    record_bytes = sum(csi_tob_datatypes(dt, bytes=True) for dt in header_info['data_types'])
    n_rec = data_block_len // record_bytes

    tag_names = list(csi_check_tagnames(header_info['field_names']))
    data = {tag: np.zeros(n_rec, dtype=csi_tob_datatypes(dt, equivalent=True))
            for tag, dt in zip(tag_names, header_info['data_types'])}

    with open(file, 'rb') as f:
        f.seek(header_info['header_bytes'])
        for r in range(n_rec):
            for tag, dt in zip(tag_names, header_info['data_types']):
                byte_count = csi_tob_datatypes(dt, bytes=True)
                data[tag][r] = np.frombuffer(f.read(byte_count), dtype=csi_tob_datatypes(dt, template=True))[0]

    if julian and 'seconds' in data:
        base_date = datetime(1990, 1, 1)
        julian_array = np.array([(base_date + timedelta(seconds=s,
                                                        microseconds=n / 1000 if 'nanosec' in data else 0)).toordinal() + 1721424.5
                                 for s, n in zip(data['seconds'], data.get('nanosec', [0] * n_rec))])
        data['JULDAY'] = julian_array

    return (data, header_info) if header else data


def csi_check_tagnames(tag_names):
    """
    Clean and make unique tag names for Campbell Scientific data fields.

    Args:
        tag_names (list): List of original tag names.

    Yields:
        str: Cleaned and unique tag names.

    Notes:
        - Converts all names to uppercase.
        - Removes illegal characters (keeps only A-Z, 0-9, and underscore).
        - Makes duplicate names unique by appending numbers.
    """
    legal_pattern = re.compile(r'[^A-Z0-9_]')
    cleaned_tags = [legal_pattern.sub('', tag.upper().strip()) or "ILLEGALCHARSONLY" for tag in tag_names]

    unique_tags = {}
    for tag in cleaned_tags:
        if tag in unique_tags:
            unique_tags[tag] += 1
            yield f"{tag}_{unique_tags[tag]}"
        else:
            unique_tags[tag] = 0
            yield tag


def csi_tob_datatypes(typestring, value=None, bytes=False, equivalent=False, template=False):
    """
    Handle data type properties and conversions for Campbell Scientific TOB formats.

    Args:
        typestring (str): TOB data type string.
        value (optional): Binary value read from file.

    Keyword Args:
        bytes (bool): If True, returns the number of bytes for this data type.
        equivalent (bool): If True, returns the Python equivalent of this data type.
        template (bool): If True, returns a NumPy dtype to read the binary data.

    Returns:
        Various: Depends on the keyword arguments set.

    Raises:
        ValueError: If an unknown data type is provided.
    """
    type_info = {
        'ieee4': (4, np.float32, float),
        'ieee4l': (4, np.float32, float),
        'ulong': (4, np.uint32, int),
        'uint4': (4, np.uint32, int),
        'long': (4, np.int32, int),
        'fp2': (2, np.dtype(('B', 2)), float)
    }

    typestring = typestring.lower()
    if typestring not in type_info:
        raise ValueError(f"Error: unknown data type in TOB file header: {typestring}")

    bytes_in_file, read_template, py_equivalent = type_info[typestring]

    if bytes:
        return bytes_in_file
    elif equivalent:
        return py_equivalent
    elif template:
        return read_template
    elif value is not None:
        return csi_fs2(value) if typestring == 'fp2' else py_equivalent(value)
    else:
        return None


def csi_fs2(two_byte_array):
    """
    Convert a two-byte array to a Campbell Scientific FS2 floating-point number.

    Args:
        two_byte_array (numpy.ndarray): Two-byte array representing an FS2 number.

    Returns:
        float: The decoded FS2 number.

    Notes:
        FS2 is a custom 2-byte floating-point format used by Campbell Scientific.
    """
    bits = np.unpackbits(np.array(two_byte_array, dtype=np.uint8))
    s, e, m = bits[0], -1.0 * (bits[2] + 2 * bits[1]), sum(bits[i] * 2 ** (15 - i) for i in range(3, 16))

    if (s, e, m) == (0, 0, 8191):
        return float('inf')
    elif (s, e, m) == (1, 0, 8191):
        return float('-inf')
    elif (s, e, m) in [(0, 0, 8190), (1, 0, 8190)]:
        return float('nan')
    else:
        return (-1.0) ** s * 10.0 ** e * m


# Example usage
if __name__ == "__main__":
    file_path = "path_to_your_tob1_file.tob1"

    try:
        # Read the TOB1 file
        data, header = csi_read_tob1(file_path, header=True, julian=True)

        # Print header information
        print("File Header:")
        for key, value in header.items():
            print(f"{key}: {value}")

        # Print first few rows of data
        print("\nFirst few rows of data:")
        for tag, values in data.items():
            print(f"{tag}: {values[:5]}")

        # Calculate and print some statistics
        print("\nData Statistics:")
        for tag, values in data.items():
            if np.issubdtype(values.dtype, np.number):
                print(f"{tag}:")
                print(f"  Mean: {np.mean(values):.2f}")
                print(f"  Min: {np.min(values):.2f}")
                print(f"  Max: {np.max(values):.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")


import struct
import os
import datetime as _dt
from operator import itemgetter

__author__ = 'spirro00'

def fp22float(fp2integer):
    inf, neginf, nan = 0x1fff, 0x9fff, 0x9ffe

    if fp2integer == inf:
        return float('inf')
    if fp2integer == neginf:
        return -float('inf')
    if fp2integer == nan:
        return float('NaN')

    mantissa, exponent = fp2integer & 0x1fff, (fp2integer & 0x6000) >> 13
    floatvalue = mantissa * 10 ** (-1. * exponent)
    if fp2integer & 0x8000:
        floatvalue *= -1
    return floatvalue


def read_cs_formats(csformat):
    pyformat = []
    knownformats = {'FP2': '>H', 'IEEE4': 'f', 'IEEE4B': '>f',
                    'UINT2': '>H', 'INT4': '>i', 'UINT4': '>L',
                    'String': 's', 'Boolean': '?', 'Bool8': '8?',
                    'LONG': 'l', 'ULONG': '>L'}
    for _ in csformat:
        if _.startswith('ASCII'):
            n_string = _.replace(')', '')
            n_string = n_string.split(sep = '(')
            pyformat.append(f'{n_string[1]}s')
        elif _ in knownformats:
            pyformat.append(knownformats[_])
        else:
            print(
                (
                    (
                        (
                            f'Warning: The format code {_}'
                            + ' is not known \n'
                            + 'please adapt the known formats (a dictionary)'
                        )
                        + 'This is done by the correct identifier from'
                    )
                    + 'https://docs.python.org/3.5/library/struct.html'
                )
            )


    return pyformat


def read_cs_files(filename, forcedatetime=False,
                  bycol=True, quiet=True, metaonly=False,**kwargs):
    with open(filename, mode = 'rb') as file_obj:
        firstline = file_obj.readline().rstrip().decode().split(sep = ',')
        firstline = [i.replace('"', '') for i in firstline]
        filetype = firstline[0]
        if '<?xml' in firstline[0]:
            # we have an xml file, and the campbell scientific xml version is given on line 2
            # shorthand is csixml
            firstline = file_obj.readline().rstrip().decode().split(sep = ',')
            firstline = [i.replace('"', '') for i in firstline]
            csixml = firstline[0][1:-1].split(' ')
            if csixml[0] != 'csixml':
                if not quiet:
                    print('Filecontent indicated XML but apparently it\'s not a csixml file')
                return False, False
            else:
                csixmlversion = float(csixml[1].split('=')[-1])
                if csixmlversion > 1.0:
                    print(
                        f'This reader has been written for CSIXML version 1.0, but the version is {csixmlversion}'
                    )

                filetype = csixml[0].upper()
        else:
            file_obj.seek(0)

        if not quiet:
            print('reading header and determening filetype')

        meta = read_cs_meta(file_obj, filetype)
        if metaonly:
            return meta
        if not quiet:
            print(f'Reading the file {filename}')

        if filetype in ['TOA5', 'TOB1', 'TOB3', 'CSIXML']:
            if not quiet:
                print(f'{filename} is a {filetype}-File')
            if filetype == 'TOA5':
                data = read_cs_toa5(file_obj,
                                    bycol = bycol, forcedatetime = forcedatetime, **kwargs)

            if filetype == 'TOB1':
                data = read_cs_tob1(file_obj, meta, **kwargs)

            if filetype == 'TOB3':
                data = read_cs_tob3(file_obj, meta, quiet = quiet, **kwargs)
                # have to insert the timestamp and recordnumber into the meta
                meta[2].insert(0, 'RECORD'), meta[2].insert(0, 'TIMESTAMP')

                # units
                meta[3].insert(0, 'RN'), meta[3].insert(0, 'TS')

                # sampled as what
                meta[4].insert(0, ' '), meta[4].insert(0, ' ')

                # corresponding units
                meta[5].insert(0, 'ULONG'), meta[5].insert(0, 'DATETIME')

            if filetype == 'CSIXML':
                data = read_cs_csixml(file_obj, bycol=bycol,
                                      forcedatetime = forcedatetime,**kwargs)

            return data, meta



        else:
            if not quiet:
                print('Neither TOA5,TOB1, TOB3 not CSIXML-File')
            return False, False


def read_cs_meta(file_obj, filetype):
    filetypes = {'TOA5': 4,
                 'TOB1': 5,
                 'TOB3': 6,
                 'CSIXML': -1,  # CSIXML is special insofar as it has "unlimited" number of headerlines
                 }
    metalines = filetypes.get(filetype, 0)
    if metalines >= 0:
        meta = [
            file_obj.readline().rstrip().decode().split(sep=',')
            for _ in range(metalines)
        ]

    elif filetype == 'CSIXML':
        import xml.etree.ElementTree as ET

        # there needs to be a opening statement like <head>
        tree = ET.parse(file_obj.name)
        root = tree.getroot()

        # we will need a nested list
        meta = [[i.text for i in root.getchildren()[0][0]]]

        # these are by default process name type, but we'd like them as name process type..
        metakeys = sorted(root.getchildren()[0][1][0].keys())

        meta.extend(
            [i.attrib[metakeys[line]] for i in root.getchildren()[0][1]]
            for line in range(len(metakeys))
        )

        # to adapt this type of file to the rest (so everything gives the same format)
        # we here add the Timestamp and Record to the meta header
        meta[1] = ['TIMESTAMP', 'RECORD',] + meta[1]
        meta[2] = [' ', ' ',] + meta[2]
        meta[3] = ['DATETIME', 'ULONG',] + meta[3]

    else:
        print('Here can follow other filetype headers..')

    for i, ii in enumerate(meta):
        meta[i] = [j.replace('"', '') for j in ii]
    return meta


def read_cs_convert_tob3_daterec(seconds):  # , milliseconds):
    # print(seconds)
    td = _dt.timedelta(seconds = seconds)  # ,milliseconds=milliseconds)
    basedate = _dt.datetime(year = 1990,
                            month = 1, day = 1, hour = 0,
                            second = 0, microsecond = 0)
    return basedate + td


def read_cs_convert_tob1_daterec(daterec):
    basedate = _dt.datetime(year = 1989, month = 12, day = 31, hour = 12)
    date = (daterec[0] + daterec[1] / 10 ** 9) / (24 * 3600)
    td = _dt.timedelta(seconds = daterec[0],
                       microseconds = daterec[1] / 10 ** 3)
    date = [basedate + td]
    date.extend(iter(daterec[2:]))
    return date


def read_cs_csixml(file_obj, bycol=True, forcedatetime=False, guesstype=False):
    import xml.etree.ElementTree as ET
    # there needs to be a opening statement like <head>
    tree = ET.parse(file_obj.name)
    root = tree.getroot()
    # we will need a nested list for the data
    # [1] contains the data
    data = []
    for record in root.getchildren()[1]:

        # the timestamp and recordnumber are in the xml tags
        entry = [record.attrib['time'], int(record.attrib['no']), ]

        # but the "float" numbers are in the text tag
        entry += [rec.text for rec in record]

        data.append(entry)

    if bycol:
        data = list(map(list, zip(*data)))

    if forcedatetime:
        tf = '%Y-%m-%dT%H:%M:%S'

        # account for float seconds, which may not be with . on every line
        ftf = ['', '.%f']

        if bycol:
            data[0] = [_dt.datetime.strptime(_, tf + ftf['.' in _]) for _ in data[0]]
        else:
            for datum in data:
                datum[0] = _dt.datetime.strptime(datum[0], tf + ftf['.' in datum[0]])



    if guesstype:
        for i in range(len(data[1:])):
            data[i + 1] = [float(j) if j.isdigit() else j for j in data[i + 1]]

    return data


def read_cs_toa5(file_obj,
                 forcedatetime=False,
                 bycol=True,
                 guesstype=False,
                 **kwargs):
    data = [i.rstrip().decode().replace('"', '').split(sep = ',') for i in file_obj]

    if bycol:
        data = list(map(list, zip(*data)))

    if forcedatetime:
        tf = '%Y-%m-%d %H:%M:%S'

        # account for float seconds, which may not be with . on every line
        ftf = ['', '.%f']

        if bycol:
            data[0] = [_dt.datetime.strptime(_, tf + ftf['.' in _]) for _ in data[0]]
        else:
            for i in range(len(data)):
                data[i][0] = _dt.datetime.strptime(data[i][0], tf + ftf['.' in data[i][0]])


        data[0] = [_dt.datetime.strptime(_, tf + ftf['.' in _]) for _ in data[0]]

    if guesstype:
        for i in range(len(data[1:])):
            data[i + 1] = [float(j) if j.isdigit() else j for j in data[i + 1]]

    return data


def read_cs_tob1(file_obj, meta,
                 bycol=True,
                 **kwargs):
    csformat = meta[-1]
    pyformat = read_cs_formats(csformat)
    #    print(csformat)
    subrecsizes = sum(struct.Struct(i).size for i in pyformat)
    recbegin = file_obj.tell()
    n_rec_total = (os.path.getsize(file_obj.name) - recbegin) / subrecsizes
    data = []
    for _ in range(int(n_rec_total)):
        tempdata = []
        for ii in pyformat:
            nbyte = struct.Struct(ii).size
            if ii == 'L':
                ii = '>L'
            tdata = struct.unpack_from(ii, file_obj.read(nbyte))[0]
            if ii == '>H':
                tdata = fp22float(tdata)
            tempdata.append(tdata)
        data.append(list(tempdata))
    for i, ii in enumerate(data):
        data[i] = read_cs_convert_tob1_daterec(ii)
    if bycol:
        data = list(map(list, zip(*data)))
    return data


def read_cs_tob3(file_obj, meta,
                 quiet=True,
                 bycol=True,
                 **kwargs
                 ):
    csformat = meta[-1]
    pyformat = read_cs_formats(csformat)
    # account for system (since the hdr is of longs of size)
    fhdrformats = ['L', 'l', 'i', 'I']
    for _ in fhdrformats:
        if struct.Struct(3 * _).size == 12:
            hdrformat = _
    fhdr, ffoot = 3 * hdrformat, 'HH'

    fhdrsize, ffootsize = struct.Struct(fhdr).size, struct.Struct(ffoot).size
    # the variables are taken from "Campbell Scientific Data File Formats"
    # by Jon Trauntvein, Thursday 13 February, 2002 Version 1.1.1.10
    tablename = meta[1][0]
    framesize = meta[1][2]  # size in bytes including frameheader and framefooter
    tablesize = meta[1][3]  # Intended table size

    ######## IMPORTANT FRAME VALIDATION #######
    # validation stamp, IMPORTANT
    validation = [int(meta[1][4])]
    # extend validation stamp, IMPORTANT
    validation.append(2 ** 16 - 1 - validation[0])

    frametimeresolution = meta[1][5]
    # since only the whole frame has a timestamp, this is the delta time for subrecs
    frameresolution = int(meta[1][1].split(sep = ' ')[0])
    multiplier = meta[1][1].split(sep = ' ')[1]

    # len > 3 gives us a scaling factor for the rest of the string
    if multiplier[0].isalpha():
        if multiplier.__len__() > 3:
            multiplier_scale_dict = {'U': 10 ** 6, 'M': 10 ** 3}
            if multiplier[0] in multiplier_scale_dict:
                prescale = multiplier_scale_dict[multiplier[0]]
                multiplier = multiplier[1:]
            else:
                print('warning, length indicates a multiplier_scale (',
                      multiplier[0],
                      '), but none found',
                      )
                prescale = 1. ** 0
        else:
            if not quiet:
                print('No multiplier_scale found')
                print('Abbreviation is only 3 letters long')
            prescale = 1. ** 0

        # should be expanded for the corrsponding amount of seconds in the mulitpliert
        time_abbr_dict = {'MIN': 60., 'SEC': 1.}
        if multiplier in time_abbr_dict:
            multiplier = prescale / time_abbr_dict[multiplier]
        else:
            multiplier = prescale / time_abbr_dict['SEC']
            print('warning, time abbreviation could not be found')
            print('Defaulting to seconds')
    else:
        print('warning, multiplier may not be correctly parsed and is set to 1')
        multiplier = 1 ** 0

    subrec_step = frameresolution / multiplier
    scale = frametimeresolution[3:].rstrip('sec')

    nscale = int(scale[:-1])
    if scale[-1].isalpha():
        if scale[-1] == 'U': scalefac = 10 ** 6
        if scale[-1] == 'M': scalefac = 10 ** 3
    else:
        scalefac = 1 ** 0
    subrec_scale = nscale / scalefac

    subrecsizes = sum(struct.Struct(i).size for i in pyformat)
    n_rec_frame = (int(framesize) - struct.Struct(fhdr + ffoot).size) // subrecsizes
    basestruct = struct.Struct(fhdr + ffoot).size + subrecsizes * n_rec_frame
    recbegin = file_obj.tell()
    filesize = os.path.getsize(file_obj.name)
    n_rec_total = (filesize - recbegin) / basestruct

    seconds, millisecs, recordnumber = [], [], []
    rec, recfoot, rechdr, timerec = [], [], [], []
    framecnt = 0

    while file_obj.tell() != filesize-fhdrsize-subrecsizes * n_rec_frame-ffootsize:


        binary_fhdr = file_obj.read(fhdrsize)

        if not binary_fhdr or len(binary_fhdr) < fhdrsize:
            # end of file reached (file_obj.read returns an emptry string)
            # the footer below should be unneeded but we leave it in case
            # the  TOB3 is strange (which it is)
            print(binary_fhdr,fhdrsize,len(binary_fhdr))
            break

        rechdr.append(struct.unpack_from(fhdr, binary_fhdr))
        inpos = file_obj.tell()
        outpos = file_obj.seek(inpos + subrecsizes * n_rec_frame)
        binary_footer = file_obj.read(ffootsize)

        if not binary_footer or len(binary_footer) < ffootsize:
            # end of file reached (file_obj.read returns an emptry string)
            break

        x = struct.unpack_from(ffoot, binary_footer)

        framecnt += 1
        if x[1] in validation:
            file_obj.seek(inpos)
            if x[0] != 0:
                # this is a minor frame

                temprec = []
                for ii in range(n_rec_frame):

                    minrec = []

                    for iii in pyformat:
                        recsize = struct.Struct(iii).size
                        one_record = struct.unpack_from(iii, file_obj.read(recsize))[0]

                        if iii == '>H':
                            one_record = fp22float(one_record)
                        if iii[-1] == 's':
                            one_record = one_record.decode('unicode_escape')
                        minrec.append(one_record)

                    y = struct.unpack_from(ffoot, file_obj.read(ffootsize))

                    if y[1] in validation:
                        if y[0] == 0:
                            minor_rec = 0
                        else:
                            offset = (bin(y[0]))
                            sizeoffset = offset[6:]  # 4+2 for the 0b
                            minor_rec = (int(sizeoffset, 2) - ffootsize - fhdrsize)
                        n_minor_rec = minor_rec // subrecsizes
                        # compare to ii+1 because the n_minor_rec is the full number
                        # whereas the ii is from the range and starts at 0
                        if n_minor_rec == (ii+1):
                            rec.extend(temprec)
                            recordnumber.extend(range(rechdr[-1][2], rechdr[-1][2] + n_minor_rec))
                            seconds.extend(
                                rechdr[-1][0] + (i * subrec_step + subrec_scale * rechdr[-1][1]) for i in
                                range(n_minor_rec))
                            file_obj.seek(outpos + ffootsize)
                            # this breaks the for loop
                            break
                    else:
                        temprec.append(minrec)
                        # +1 on ii because at the end of this loopiteration
                        # the ii is not yet increased but we read the record
                        # and need to move on further
                        file_obj.seek(inpos + (ii+1) * subrecsizes)
            else:
                # this is a major frame, easy
                for _ in range(n_rec_frame):
                    temprec = []
                    for iii in pyformat:
                        one_record = struct.unpack_from(iii, file_obj.read(struct.Struct(iii).size))[0]
                        if iii[-1] == 's':
                            one_record = one_record.decode('unicode_escape')
                        if iii == '>H':
                            one_record = fp22float(one_record)
                        temprec.append(one_record)

                    rec.append(temprec)
                recordnumber.extend(range(rechdr[-1][2], rechdr[-1][2] + n_rec_frame))
                seconds.extend(
                    rechdr[-1][0] + (i * subrec_step + subrec_scale * rechdr[-1][1]) for i in
                    range(n_rec_frame))
                file_obj.seek(outpos + ffootsize)
    timestamp = [
        read_cs_convert_tob3_daterec(seconds[ii])
        for ii, i in enumerate(seconds)
    ]

    for i, ii in enumerate(rec):
        rec[i].insert(0, recordnumber[i])
        rec[i].insert(0, timestamp[i])

    rec.sort(key = itemgetter(1))
    if bycol:
        rec = list(map(list, zip(*rec)))

    return rec

