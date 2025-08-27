# -----------------------------------------------------------------------------
# Read wikidata source and create data file of names.
# Example usage:
# pbzip2 -d -c -m200 /data/wikidata-20250526-all.json.bz2| python extract_wikidata_names.py -i - -o 20250526_names_wikidata.csv 2> 20250526_err.out
# Output format:
#   wikidata id,canonical name,language code,name
#
# Download Wikidata from: https://dumps.wikimedia.org/wikidatawiki/entities/
# Note: the download is large and takes quite some time,
#   so it's best to download from a dated directory
# EG:
# curl --retry 9 -C - -L -R -O https://dumps.wikimedia.org/wikidatawiki/entities/20250707/wikidata-20250707-all.json.bz2
#

# -----------------------------------------------------------------------------
import argparse
import bz2
import concurrent.futures
import csv
import gzip
import itertools
import sys
import time
import traceback
import unicodedata
from datetime import datetime

import orjson

# -----------------------------------------------------------------------------
# constants
# -----------------------------------------------------------------------------

file_path = ""
debug_on = True
number_of_lines_to_process = 0
number_of_names_to_process = 0
status_print_lines = 10000


# =============================================================================
def debug(text):
    if debug_on:
        print(text, file=sys.stderr, flush=True)


# =============================================================================
def format_seconds_to_hhmmss(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


# =============================================================================
# get the canonical name for this entity
def getAsciiName(entity, name_field):
    names = entity["NAMES"]
    if not names:
        return ""
    # print(names[0])
    en_name = names[0].get(name_field, "")
    # print(f"--{en_name}")

    # normalize any other non-ascii characters
    en_name = unicodedata.normalize('NFKC', en_name).encode('ascii', 'ignore').decode('ascii')
    # debug(f'>>return:{entity["id"]}: {en_name.lower()}')
    return en_name.lower()


# =============================================================================
# process with black listed properties
def process_json_line_whitelisted(line, record_type):
    # Parse the JSON string
    entity = orjson.loads(line)
    # out_string = f'{entity["RECORD_ID"]} : {entity["RECORD_TYPE"]}'
    if entity["RECORD_TYPE"] != record_type:
        return

    name_field = ""
    if entity["RECORD_TYPE"] == "PERSON":
        name_field = "NAME_FULL"
    elif entity["RECORD_TYPE"] == "ORGANIZATION":
        name_field = "NAME_ORG"
    file_output_dict = {}

    # get the english name
    en_name = entity["RECORD_ID"].lower()
    # en_name = getAsciiName(entity, name_field)
    # if en_name == "":
    #     debug(f'!! No canonical name for {entity["id"]}')
    #     return
    # out_string += f' : {en_name}(en)'

    names = entity["NAMES"]
    if not names:
        return
    for n in names:
        name = n[name_field]
        if not name:
            debug(f"Name field not found: {n}")
        else:
            file_output_dict[name] = [entity["RECORD_ID"], en_name, "unknown"]
            # print(file_output_dict[name])
    # print(f">> {en_name}", flush=True)
    return file_output_dict


# =============================================================================
# process a line from the file
def process_line(line, record_type):
    stripped_line = line.strip()
    if "" == stripped_line or len(stripped_line) < 10:
        return
    # strip off the comma
    if stripped_line.endswith(","):
        stripped_line = stripped_line[:-1]
    return process_json_line_whitelisted(stripped_line, record_type)


# =============================================================================
# Read a jsonl file and process concurrently
def read_file_futures(file_handle, output_file_path, record_type):
    line_count = 0
    name_count = 0
    shutdown = False
    start_time = time.time()
    with open(output_file_path, "w") as cjk_out:
        writer = csv.writer(cjk_out)
        writer.writerow(['id', 'canonical', 'language', 'name'])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_line, line, record_type): line
                for line in itertools.islice(file_handle, executor._max_workers * 10)
            }

            while futures:
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f in done:
                    try:
                        file_output_dict = f.result()
                        if isinstance(file_output_dict, dict) and file_output_dict:
                            if id is not None:
                                name_count += 1
                                for key, value in file_output_dict.items():
                                    writer.writerow([*value, key])
                    except Exception:
                        pass
                    else:
                        if not shutdown:
                            line = file_handle.readline()
                            if line:
                                futures[executor.submit(process_line, line, record_type)] = (line)
                                line_count += 1
                            if line_count % status_print_lines == 0:
                                print(f"☺ {line_count:,} lines read in {format_seconds_to_hhmmss(time.time() - start_time)}", flush=True, end='\r')
                            if number_of_lines_to_process > 0 and line_count >= number_of_lines_to_process:
                                executor.shutdown(wait=True, cancel_futures=False)
                                shutdown = True
                            if number_of_names_to_process > 0 and name_count >= number_of_names_to_process:
                                executor.shutdown(wait=True, cancel_futures=False)
                                shutdown = True
                    finally:
                        del futures[f]
    print("\n")
    print(f"{line_count:,} total lines read", flush=True)
    print(f"{name_count:,} total names found", flush=True)
    print(f"{format_seconds_to_hhmmss(time.time() - start_time)} total time", flush=True)
    return line_count


# =============================================================================
# Read a jsonl file and process
def read_file(file_handle, output_file_path, record_type):
    line_count = 0
    name_count = 0
    start_time = time.time()
    with open(output_file_path, "w") as cjk_out:
        writer = csv.writer(cjk_out)
        writer.writerow(['id', 'canonical', 'language', 'name'])
        for line in file_handle:
            line_count += 1
            try:
                file_output_dict = process_line(line, record_type)
                if isinstance(file_output_dict, dict) and file_output_dict:
                    if id is not None:
                        name_count += 1
                        for key, value in file_output_dict.items():
                            writer.writerow([*value, key])
            except Exception:
                pass
            if line_count % status_print_lines == 0:
                print(f"☺ {line_count:,} lines read in {format_seconds_to_hhmmss(time.time() - start_time)}", flush=True, end='\r')
            if number_of_lines_to_process > 0 and line_count >= number_of_lines_to_process:
                break
            if number_of_names_to_process > 0 and name_count >= number_of_names_to_process:
                break
    print("\n")
    print(f"{line_count:,} total lines read", flush=True)
    print(f"{name_count:,} total names found", flush=True)
    print(f"{format_seconds_to_hhmmss(time.time() - start_time)} total time", flush=True)
    return line_count


# =============================================================================
# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="extract_open_sactions_names", description="Creates CSV of names from Open Sanctions Senzing JSON file."
    )

    parser.add_argument("-i", "--infile", action="store", required=True, help='Path to Open Sactions Senzing JSON file.')
    parser.add_argument("-o", "--outfile", action="store", required=True, help='Path to output CSV file.')
    parser.add_argument("-t", "--record_type", action="store", required=True, help='PERSON or ORGANIZATION')
    args = parser.parse_args()

    infile_path = args.infile
    outfile_path = args.outfile
    line_count = 0
    start_time = datetime.now()
    print("\n")
    print("☺", flush=True, end='\r')

    try:
        if infile_path.endswith(".bz2"):
            debug(f"Opening {infile_path}...")
            with bz2.open(infile_path, 'rt') as f:
                line_count = read_file_futures(f, outfile_path, args.record_type)
        elif infile_path.endswith(".gz"):
            debug(f"Opening {infile_path}...")
            with gzip.open(infile_path, 'rt') as f:
                line_count = read_file_futures(f, outfile_path, args.record_type)
        elif infile_path.endswith(".json") or infile_path.endswith(".jsonl"):
            debug(f"Opening {infile_path}...")
            with open(infile_path, 'rt') as f:
                line_count = read_file(f, outfile_path, args.record_type)
        elif infile_path == "-":
            debug("Opening stdin...")
            with open(sys.stdin.fileno(), 'rt') as f:
                line_count = read_file_futures(f, outfile_path, args.record_type)
        else:
            debug("Unrecognized file type.")
    except Exception:
        traceback.print_exc()

    end_time = datetime.now()
    print(f"Input read from {infile_path}, output written to {outfile_path}.", flush=True)
    print(f"    Started at: {start_time}, ended at: {end_time}.", flush=True)
    print("\n-----------------------------------------\n")

# pbzip2 -d -c -m200 /data/wikidata-20250714-all.json.bz2| python extract_wikidata_names.py -i - -o ./data/20250714_names_wikidata.csv 2> 20250714_err.out
