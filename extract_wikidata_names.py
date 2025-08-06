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

import script_classifier as script

# -----------------------------------------------------------------------------
# constants
# -----------------------------------------------------------------------------

file_path = ""
debug_on = True
number_of_lines_to_process = 0
number_of_names_to_process = 0
status_print_lines = 10000

BLACKLIST_CANDIDATES = {}

INSTANCE_OF_BLACKLIST = {
    "Q171": "wiki",
    "Q7397": "software",
    "Q40614": "fossil",
    "Q41298": "magazine",
    "Q41710": "ethnic group",
    "Q43229": "organization",
    "Q43616": "mummy",
    "Q163740": "nonprofit organization",
    "Q215380": "musical group",
    "Q474968": "anonymous master",
    "Q1052390": "most recent common ancestor",
    "Q1412596": "simulacrum",
    "Q1747829": "notname",
    "Q2707384": "side project",
    "Q4233718": "anonymous",
    "Q10648343": "duo",
    "Q10855061": "archaeological find",
    "Q12737077": "occupation",
    "Q13442814": "scholarly article",
    "Q13473501": "collective",
    "Q15097084": "heritage register",
    "Q18347143": "Hominin fossil",
    "Q18356450": "unidentified decedent",
    "Q105525662": "human remains",
    "Q106892475": "research group",
    "Q119579738": "M-8-1",
    "Q17558136": "YouTube channel",
    "Q16334295": "group of humans",
    "Q1241025": "research group",
    "Q673899": "study group",
    "Q4830453": "business",
    "Q14946528": "conflation",
    "Q2088357": "musical ensembel",
    "Q281643": "musical trio",
    "Q55753593": "sibling trio",
    "Q1165905": "sexual identity",
    "Q783794": "company",
    "Q2927074": "internet meme",
    "Q9621": "human skeleton",
    "Q199414": "bog body",
    "Q216353": "title",
    "Q7558495": "solo musical project",
    "Q7881": "skeleton",
    "Q839954": "archaeological site",
    "Q220659": "archaeological artefact",
    "Q26513": "human fetus",
    "Q1656682": "event",
    "Q8436": "family",
    "Q16519632": "scientific organization",
    "Q4164871": "position",
    "Q968159": "art movement",
    "Q486972": "human settlement",
    "Q10737": "suicide",
    "Q1318274": "placeholder name",
    "Q11664239": "music unit",
    "Q1077857": "persona",
    "Q132821": "murder",
    "Q844482": "killing",
    "Q814254": "feature",
    "Q35779580": "possibly invalid entry requiring further references",
    "Q1190554": "occurrence",
    "Q72398691": "video game news website",
    "Q3235597": "crucifixion",
    "Q47461344": "written work",
    "Q1792356": "art book publisher",
    "Q12888920": "selection",
    "Q532": "village",
    "Q109115381": "blockchain game",
    "Q35127": "website",
    "Q24238356": "unknown",
    "Q9212979": "musical duo",
    # "Q60539479": "positive emotion",
    # "Q331769": "mood",
    # "Q41537118": "emotional state",
    # "Q3968640": "mental state",
    # "Q130459448": "condition type",
    # "Q15632617": "fictional human",
    # "Q15773317": "television character",
    # "Q20085850": "fictional vigilante",
    # "Q277759": "book series",
    # "Q106974458": "long-series books",
    # "Q712378": "organ",
    # "Q112826905": "class of anatomical entity",
    # "Q59541917": "Wikimedia topic category",
    # "Q4167836": "Wikimedia category",
    # "Q17362920": "Wikimedia duplicated page",
    # "Q11266439": "Wikimedia template",
    # "Q17146139": "Wikimedia route diagram",
    # "Q47150325": "calendar day of a given year",
    # "Q67131190": "Wikimedia tracking category",
    # "Q13442814": "scholarly article",
    # "Q94574287": "Wikinews date page",
    # "Q29964144": "year BC",
    # "Q577": "year",
    # "Q3186692": "calendar year",
    # "Q235670": "common year starting and ending on Sunday",
    # "Q36330215": "Wikimedia location map template",
    # "Q4820": "Template:New Wave (Contest)",
    # "Q11753321": "Wikimedia navigational template",
    # "Q19842659)": "Wikimedia user language template",
    # "Q108783631": "Wikimedia country data template",
    # "Q19887878": "Wikimedia infobox template",
    # "Q15184295": "Wikimedia module",
    # "Q115595777": "taxonomy template",
    # "Q116313869": "Wikimedia family tree template",
    # "Q14204246": "Wikimedia project page",
    # "Q116152698": "Wikimedia subtemplate",
    # "Q21286738": "Wikimedia permanent duplicate item",
    # "Q97303168": "Wikimedia deletion template",
    # "Q110010043": "Wikimedia copyright template",
    # "Q107285679": "Wikimedia navigational template for sports team squad",
    # "Q112869585": "Wikimedia stub template",
    # "Q621080": "Library of Congress Classification",
    # "Q15101896": "asteroid classification",
    # "Q30432511": "Wikimedia meta category",
    # "Q15647814": "Wikimedia administration category",
    # "Q23894233": "Wikimedia templates category",
    # "Q56428020": "Wikimedia lists category",
    # "Q28123792": "German pregnancy category",
    # "Q30330522": "Wikimedia unknown parameters category",
    # "Q15407973": "Wikimedia disambiguation category",
    # "Q125101059": "Wikimedia soft redirect category",
    # "Q20010800": "Wikimedia user language category",
    # "Q24046192": "Wikimedia category of stubs",
    # "Q13331174": "Wikimedia navboxes category",
    # "Q106575300": "Wikimedia albums-by-genre category",
    # "Q125101257": "Wikimedia category of redirects",
    # "Q106612246": "Wikimedia albums-by-performer category",
    # "Q13406463": "Wikimedia list article",
    # "Q107344376": "Wikimedia module configuration",
    # "Q116152754": "Wikimedia submodule",
    # "Q18340514": "events in a specific year or time period",
    # "Q17442446": "Wikimedia internal item",
    # "Q4663903": "Wikimedia portal",
    # "Q4656150": "Wikimedia project policies and guidelines",
    # "Q66715753": "Wikimedia list of persons by position held",
    # "Q29654788": "Unicode character",
}

INSTANCE_OF_WHITELIST = {
    "Q5": "human",
}


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
def getAsciiName(entity):
    en_name = entity.get("labels", {}).get("en", {}).get("value", "")
    # if the 'en' label doesn't exist look for other forms of an 'en' label
    if len(en_name.strip()) == 0:
        en_name = entity.get("labels", {}).get("en-gb", {}).get("value", "")
        # debug(f'>>en-gb:{entity["id"]}: {en_name}')
    if len(en_name.strip()) == 0:
        en_name = entity.get("labels", {}).get("en-ca", {}).get("value", "")
        # debug(f'>>en-ca:{entity["id"]}: {en_name}')
    # if the label doesn't exist then look for a sitelink title
    if len(en_name.strip()) == 0:
        # debug(f'>>sitelink:{entity["id"]}:{entity.get("sitelinks", {}).get("enwiki", {}).get("title", "")}')
        en_name = entity.get("sitelinks", {}).get("enwiki", {}).get("title", "")
    # if the label and sitelink title don't exist, look for an en alias
    if len(en_name.strip()) == 0:
        # debug(f'>>{entity["id"]}:{entity.get("aliases", {}).get("en", {})[0].get("value", "")}')
        alias_list = entity.get("aliases", {}).get("en", {})
        if alias_list:
            en_name = alias_list[0].get("value", "")
    # bail when we still haven't found an English name to work with
    if len(en_name.strip()) == 0:
        return ""

    # see if there is already an ascii name alias
    if not script.wordIsIn(en_name, script.ascii):
        en_aliases = entity.get("aliases", {}).get("en", {})
        for alias in en_aliases:
            val = alias.get("value")
            if val and script.wordIsIn(val, script.ascii):
                # debug(f'>>return alias:{entity["id"]}: {val}')
                return val.lower()

    # normalize any other non-ascii characters
    en_name = unicodedata.normalize('NFKD', en_name).encode('ascii', 'ignore').decode('ascii')
    # debug(f'>>return:{entity["id"]}: {en_name.lower()}')
    return en_name.lower()


# =============================================================================
# process with black listed properties
def process_json_line_whitelisted(line):
    # Parse the JSON string
    entity = orjson.loads(line)
    out_string = f'{entity["id"]} : {entity["type"]}'
    file_output_dict = {}
    file_should_write = False
    # get the english name
    en_name = getAsciiName(entity)
    if en_name == "":
        debug(f'!! No canonical name for {entity["id"]}')
        return
    instance_of = entity.get("claims", {}).get("P31")
    out_string += f' : {en_name}(en)'
    if instance_of:
        ids = []
        blacklisted = False
        for inst in instance_of:
            prop_id = inst.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
            if prop_id is not None:
                ids.append(prop_id)
        for prop_id in ids:
            prop_value = INSTANCE_OF_BLACKLIST.get(prop_id)
            if prop_value:
                out_string += f' blacklisted[for {prop_value} {prop_id}] all: {ids}'
                blacklisted = True
        if not blacklisted:
            for prop_id in ids:
                prop_value = INSTANCE_OF_WHITELIST.get(prop_id)
                if prop_value:
                    out_string += f' whitelisted[for {prop_value} {prop_id}] all: {ids}'
                    file_should_write = True
                    if len(ids) > 1:
                        out_string += f' all props: {ids}'
                        BLACKLIST_CANDIDATES[entity["id"]] = (en_name, ids)
                        # print(f"{entity['id']} ->   {ids}")
                    break
    else:
        # require that each item be an instance of something.
        # debug(f'no instance of for {entity["id"]}')
        return
    if file_should_write:
        # look for labels to add
        if entity["labels"]:
            for lang in entity["labels"]:
                lang_prop = entity["labels"][lang]
                local_name = lang_prop["value"]
                file_output_dict[local_name] = [entity["id"], en_name, lang]
                #   out_string += f' : {local_name}({lang_prop["language"]})[{rune_count}]'
        # look for aliases to add
        if entity["aliases"]:
            for lang in entity["aliases"]:
                lang_prop = entity["aliases"][lang]
                for prop in lang_prop:
                    local_name = prop["value"]
                    file_output_dict[local_name] = [entity["id"], en_name, lang]
                    #   out_string += f' : {local_name}({prop["language"]})[{rune_count}]'
        nicknames = entity.get("claims", {}).get("P1449")
        if nicknames:
            for nickname in nicknames:
                nick = nickname.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("text")
                lang = nickname.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("language")
                file_output_dict[nick] = [entity["id"], en_name, lang]
        debug(out_string)
        return file_output_dict


# =============================================================================
# process a line from the file
def process_line(line):
    stripped_line = line.strip()
    if "" == stripped_line or len(stripped_line) < 10:
        return
    # strip off the comma
    if stripped_line.endswith(","):
        stripped_line = stripped_line[:-1]
    return process_json_line_whitelisted(stripped_line)


# =============================================================================
# Read a jsonl file and process concurrently
def read_file_futures(file_handle, output_file_path):
    line_count = 0
    name_count = 0
    shutdown = False
    start_time = time.time()
    with open(output_file_path, "w") as cjk_out:
        writer = csv.writer(cjk_out)
        writer.writerow(['id', 'canonical', 'language', 'name'])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_line, line): line
                for line in itertools.islice(file_handle, executor._max_workers * 10)
            }

            while futures:
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f in done:
                    try:
                        file_output_dict = f.result()
                        if file_output_dict is not None:
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
                                futures[executor.submit(process_line, line)] = (line)
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
def read_file(file_handle, output_file_path):
    line_count = 0
    name_count = 0
    start_time = time.time()
    with open(output_file_path, "w") as cjk_out:
        writer = csv.writer(cjk_out)
        for line in file_handle:
            try:
                file_output_dict = process_line(line)
                if file_output_dict is not None:
                    id = file_output_dict["id"]
                    if id is not None:
                        name_count += 1
                        del file_output_dict["id"]
                        for key, value in file_output_dict.items():
                            writer.writerow([id, key, *value])
            except Exception:
                pass
            if line_count % status_print_lines == 0:
                print(f"{line_count:,} lines read in {format_seconds_to_hhmmss(time.time() - start_time)} seconds", flush=True)
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
        prog="extract_wikidata_names", description="Creates CSV file personal names."
    )

    parser.add_argument("-i", "--infile", action="store", required=True)
    parser.add_argument("-o", "--outfile", action="store", required=True)
    # parser.add_argument("infile", type=str, help="File of lines to phrase")
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
                line_count = read_file_futures(f, outfile_path)
        elif infile_path.endswith(".gz"):
            debug(f"Opening {infile_path}...")
            with gzip.open(infile_path, 'rt') as f:
                line_count = read_file_futures(f, outfile_path)
        elif infile_path == "-":
            debug("Opening stdin...")
            with open(sys.stdin.fileno(), 'rt') as f:
                line_count = read_file_futures(f, outfile_path)
        else:
            debug("Unrecognized file type.")
    except Exception:
        traceback.print_exc()

    end_time = datetime.now()
    print(f"Input read from {infile_path}, output written to {outfile_path}.", flush=True)
    print(f"    Started at: {start_time}, ended at: {end_time}.", flush=True)
    print("\n-----------------------------------------\n")
    print("Blacklist candidates:\n")
    for k, v in BLACKLIST_CANDIDATES.items():
        print(f"{k}: {v}")

# pbzip2 -d -c -m200 /data/wikidata-20250714-all.json.bz2| python extract_wikidata_names.py -i - -o ./data/20250714_names_wikidata.csv 2> 20250714_err.out
