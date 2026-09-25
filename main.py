import os
import re
import sqlite3
import unicodedata
import time
from collections import defaultdict

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "train"
TEST_DIR = "test"

TRAIN_SOURCE1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_SOURCE2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_SOURCE3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
GROUND_TRUTH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_SOURCE1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_SOURCE2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_SOURCE3 = os.path.join(TEST_DIR, "test_source3.tsv")

OUTPUT_DIR = "output"

TRAIN_DB = "train_blocking.db"
TEST_DB = "test_blocking.db"

TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "train_candidates.tsv")
TEST_OUTPUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

CHUNK_SIZE = 50000

# Blocks larger than this are reported.
# They are NOT automatically removed because recall is the priority.
LARGE_BLOCK_LIMIT = 10000

REQUIRED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country"
]


# ============================================================
# NORMALIZATION
# ============================================================

ABBREVIATIONS = {
    "pvt": "private",
    "pvt.": "private",
    "ltd": "limited",
    "ltd.": "limited",
    "inc": "incorporated",
    "inc.": "incorporated",
    "corp": "corporation",
    "corp.": "corporation",
    "co": "company",
    "co.": "company",
    "llc": "limitedliabilitycompany",
    "llp": "limitedliabilitypartnership",
    "plc": "publiclimitedcompany",
    "intl": "international",
    "int'l": "international",
}

STOPWORDS = {
    "the",
    "and",
    "of",
    "for",
    "a",
    "an",
    "at",
    "in",
    "on",
    "to"
}


def safe_string(value):
    """
    Convert NaN/None to an empty string.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value)


def normalize_text(value):
    """
    General normalization for business names and addresses.

    Keeps meaningful letters and numbers.
    Removes punctuation.
    Handles Unicode normalization and accents.
    """

    text = safe_string(value)

    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase / case folding
    text = text.casefold()

    # Replace common symbols
    text = text.replace("&", " and ")

    # Convert accented Latin characters to ASCII where possible.
    # Other Unicode scripts are retained.
    decomposed = unicodedata.normalize("NFKD", text)

    text = "".join(
        char
        for char in decomposed
        if not unicodedata.combining(char)
    )

    # Expand conservative abbreviations
    for old, new in ABBREVIATIONS.items():
        text = re.sub(
            r"\b" + re.escape(old) + r"\b",
            new,
            text
        )

    # Keep Unicode letters/numbers.
    # Everything else becomes a space.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_country(value):
    """
    Normalize country separately.
    """
    text = normalize_text(value)

    country_map = {
        "usa": "unitedstates",
        "us": "unitedstates",
        "u s": "unitedstates",
        "uk": "unitedkingdom",
        "u k": "unitedkingdom",
        "uae": "unitedarabemirates",
        "u a e": "unitedarabemirates",
    }

    return country_map.get(text, text)


# ============================================================
# TOKEN / ADDRESS HELPERS
# ============================================================

def get_tokens(text):
    if not text:
        return []

    return [
        token
        for token in text.split()
        if token and token not in STOPWORDS
    ]


def get_meaningful_tokens(text):
    """
    Keep tokens that are reasonably informative.

    Numbers are retained because house numbers, postal codes,
    branch numbers, etc. can be important.
    """

    tokens = get_tokens(text)

    result = []

    for token in tokens:
        if len(token) >= 2 or token.isdigit():
            result.append(token)

    return result


def extract_postcode(address):
    """
    Generic postal-code extraction.

    This is intentionally conservative because postal formats
    differ between countries.
    """

    address = safe_string(address)

    if not address:
        return ""

    patterns = [
        r"\b\d{5,6}\b",
        r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b",
        r"\b\d{4}\s?\d{3}\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            address,
            flags=re.IGNORECASE
        )

        if match:
            return normalize_text(match.group(0))

    return ""


def extract_house_number(address):
    """
    Extract the first plausible house/building number.
    """

    address = safe_string(address)

    if not address:
        return ""

    match = re.search(
        r"^\s*([0-9]+[A-Za-z]?)\b",
        address
    )

    if match:
        return normalize_text(match.group(1))

    return ""


def extract_street(address):
    """
    Lightweight street component.

    We do NOT assume that the final address words are always
    city/state because international addresses vary greatly.
    """

    text = normalize_text(address)

    if not text:
        return ""

    parts = text.split(",")

    if len(parts) >= 2:
        return parts[0].strip()

    tokens = text.split()

    if len(tokens) >= 3:
        return " ".join(tokens[:4])

    return text


def extract_address_components(address):
    """
    Extract useful but conservative address components.
    """

    raw = safe_string(address)

    normalized = normalize_text(raw)

    postcode = extract_postcode(raw)
    house_number = extract_house_number(normalized)
    street = extract_street(raw)

    return {
        "postcode": postcode,
        "house_number": house_number,
        "street": street,
    }


# ============================================================
# PHONETIC HELPER
# ============================================================

def simple_soundex(word):
    """
    Simple English-oriented Soundex.

    Used only as an additional blocking key.
    It is NOT the primary matching method.
    """

    word = safe_string(word)

    if not word:
        return ""

    word = word.upper()

    first = word[0]

    if not first.isalpha():
        return ""

    mapping = {
        "B": "1",
        "F": "1",
        "P": "1",
        "V": "1",

        "C": "2",
        "G": "2",
        "J": "2",
        "K": "2",
        "Q": "2",
        "S": "2",
        "X": "2",
        "Z": "2",

        "D": "3",
        "T": "3",

        "L": "4",

        "M": "5",
        "N": "5",

        "R": "6",
    }

    encoded = []

    previous = ""

    for char in word[1:]:

        code = mapping.get(char, "")

        if code != previous and code:
            encoded.append(code)

        previous = code

    result = first + "".join(encoded)

    return (result + "000")[:4]


# ============================================================
# BLOCK KEY GENERATION
# ============================================================

def add_key(keys, key):
    """
    Add a block key if valid.
    """

    if key:
        keys.add(key)


def generate_block_keys(
    name,
    address,
    country
):
    """
    Generate multiple high-recall blocking keys.

    Important:
    These are candidate-generation keys only.
    They are NOT final matching decisions.
    """

    keys = set()

    name = normalize_text(name)
    address = normalize_text(address)
    country = normalize_country(country)

    if not name and not address:
        return []

    name_tokens = get_meaningful_tokens(name)

    address_components = extract_address_components(address)

    postcode = address_components["postcode"]
    house_number = address_components["house_number"]
    street = address_components["street"]

    # --------------------------------------------------------
    # 1. Country + full normalized name
    # --------------------------------------------------------

    if country and name:
        add_key(
            keys,
            f"CN:{country}:{name}"
        )

    # --------------------------------------------------------
    # 2. Country + sorted name tokens
    # --------------------------------------------------------

    if country and len(name_tokens) >= 1:

        sorted_tokens = " ".join(
            sorted(set(name_tokens))
        )

        add_key(
            keys,
            f"CT:{country}:{sorted_tokens}"
        )

    # --------------------------------------------------------
    # 3. Country + first token
    # --------------------------------------------------------

    if country and name_tokens:

        add_key(
            keys,
            f"CFT:{country}:{name_tokens[0]}"
        )

    # --------------------------------------------------------
    # 4. Country + second token
    # --------------------------------------------------------

    if country and len(name_tokens) >= 2:

        add_key(
            keys,
            f"CST:{country}:{name_tokens[1]}"
        )

    # --------------------------------------------------------
    # 5. Country + first two tokens
    # --------------------------------------------------------

    if country and len(name_tokens) >= 2:

        add_key(
            keys,
            f"C2T:{country}:{name_tokens[0]}:{name_tokens[1]}"
        )

    # --------------------------------------------------------
    # 6. Country + last two tokens
    # --------------------------------------------------------

    if country and len(name_tokens) >= 2:

        add_key(
            keys,
            f"CL2:{country}:{name_tokens[-2]}:{name_tokens[-1]}"
        )

    # --------------------------------------------------------
    # 7. Position-independent token blocks
    # --------------------------------------------------------

    if country:

        for token in set(name_tokens):

            if len(token) >= 4:

                add_key(
                    keys,
                    f"TOK:{country}:{token}"
                )

    # --------------------------------------------------------
    # 8. Name prefix
    # --------------------------------------------------------

    if country and name:

        prefix = re.sub(
            r"\s+",
            "",
            name
        )[:5]

        if len(prefix) >= 3:

            add_key(
                keys,
                f"NP:{country}:{prefix}"
            )

    # --------------------------------------------------------
    # 9. Soundex blocks
    # --------------------------------------------------------

    if country and name_tokens:

        sx1 = simple_soundex(name_tokens[0])

        if sx1:
            add_key(
                keys,
                f"SX1:{country}:{sx1}"
            )

        if len(name_tokens) >= 2:

            sx2 = simple_soundex(name_tokens[1])

            if sx2:

                add_key(
                    keys,
                    f"SX2:{country}:{sx1}:{sx2}"
                )

    # --------------------------------------------------------
    # 10. Postcode + name
    # --------------------------------------------------------

    if postcode and name:

        add_key(
            keys,
            f"PCN:{postcode}:{name}"
        )

    # Postcode + first meaningful token
    if postcode and name_tokens:

        add_key(
            keys,
            f"PCT:{postcode}:{name_tokens[0]}"
        )

    # --------------------------------------------------------
    # 11. House number + name
    # --------------------------------------------------------

    if house_number and name:

        add_key(
            keys,
            f"HN:{house_number}:{name}"
        )

    # House number + first two tokens
    if house_number and len(name_tokens) >= 2:

        add_key(
            keys,
            f"H2:{house_number}:{name_tokens[0]}:{name_tokens[1]}"
        )

    # --------------------------------------------------------
    # 12. Street + name
    # --------------------------------------------------------

    if street and name:

        add_key(
            keys,
            f"STN:{street}:{name}"
        )

    # Street + first token
    if street and name_tokens:

        add_key(
            keys,
            f"STT:{street}:{name_tokens[0]}"
        )

    # --------------------------------------------------------
    # 13. Address-only keys
    # --------------------------------------------------------
    # These are useful when country/name is missing.
    # They are deliberately more conservative.

    if postcode:

        add_key(
            keys,
            f"PC:{postcode}"
        )

    if house_number and street:

        add_key(
            keys,
            f"HS:{house_number}:{street}"
        )

    return list(keys)


# ============================================================
# SQLITE SETUP
# ============================================================

def create_database(db_path):

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(
        db_path
    )

    cursor = conn.cursor()

    # Faster bulk loading while keeping a reasonable level
    # of safety for a rebuildable intermediate database.
    cursor.execute(
        "PRAGMA journal_mode=WAL"
    )

    cursor.execute(
        "PRAGMA synchronous=NORMAL"
    )

    cursor.execute(
        "PRAGMA temp_store=MEMORY"
    )

    cursor.execute(
        """
        CREATE TABLE records (
            rec_id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            entity_id TEXT,
            business_name TEXT,
            business_address TEXT,
            country TEXT
        )
        """
    )

    # One block assignment per record.
    # This prevents duplicate (block_key, rec_id) rows.
    cursor.execute(
        """
        CREATE TABLE blocks (
            block_key TEXT NOT NULL,
            rec_id INTEGER NOT NULL,
            PRIMARY KEY (block_key, rec_id)
        )
        """
    )

    cursor.execute(
        """
        CREATE INDEX idx_blocks_key
        ON blocks(block_key)
        """
    )

    cursor.execute(
        """
        CREATE INDEX idx_records_entity
        ON records(entity_id)
        """
    )

    cursor.execute(
        """
        CREATE INDEX idx_records_source
        ON records(source)
        """
    )

    conn.commit()

    return conn


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_columns(path):

    df = pd.read_csv(
        path,
        sep="\t",
        nrows=5,
        dtype=str,
        keep_default_na=False
    )

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"\nFile: {path}\n"
            f"Missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


# ============================================================
# INDEX ONE SOURCE
# ============================================================

def index_source_file(
    conn,
    path,
    source_name,
    next_rec_id
):
    """
    Index one source file.

    next_rec_id is passed in and returned so that IDs remain
    globally unique across Source 2 and Source 3.
    """

    print()
    print("=" * 70)
    print(f"INDEXING {source_name}")
    print(path)
    print("=" * 70)

    validate_columns(path)

    total_records = 0
    total_block_assignments = 0

    start_time = time.time()

    cursor = conn.cursor()

    record_batch = []
    block_batch = []

    for chunk_number, df in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        for row in df.itertuples(index=False):

            entity_id = safe_string(
                getattr(row, "entity_id", "")
            )

            business_name = safe_string(
                getattr(row, "business_name", "")
            )

            business_address = safe_string(
                getattr(row, "business_address", "")
            )

            country = safe_string(
                getattr(row, "country", "")
            )

            rec_id = next_rec_id

            next_rec_id += 1

            record_batch.append(
                (
                    rec_id,
                    source_name,
                    entity_id,
                    business_name,
                    business_address,
                    country
                )
            )

            block_keys = generate_block_keys(
                business_name,
                business_address,
                country
            )

            # Deduplicate within the current record
            for block_key in set(block_keys):

                block_batch.append(
                    (
                        block_key,
                        rec_id
                    )
                )

            total_records += 1
            total_block_assignments += len(
                set(block_keys)
            )

        # ----------------------------------------------------
        # Bulk insert records
        # ----------------------------------------------------

        cursor.executemany(
            """
            INSERT INTO records (
                rec_id,
                source,
                entity_id,
                business_name,
                business_address,
                country
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            record_batch
        )

        # ----------------------------------------------------
        # Bulk insert block assignments
        # ----------------------------------------------------

        cursor.executemany(
            """
            INSERT OR IGNORE INTO blocks (
                block_key,
                rec_id
            )
            VALUES (?, ?)
            """,
            block_batch
        )

        conn.commit()

        record_batch.clear()
        block_batch.clear()

        elapsed = time.time() - start_time

        print(
            f"{source_name} | "
            f"chunk {chunk_number} | "
            f"records={total_records:,} | "
            f"blocks={total_block_assignments:,} | "
            f"time={elapsed:.1f}s"
        )

    print()
    print(
        f"Finished {source_name}: "
        f"{total_records:,} records"
    )

    print(
        f"Block assignments: "
        f"{total_block_assignments:,}"
    )

    return next_rec_id


# ============================================================
# BLOCK SIZE ANALYSIS
# ============================================================

def analyze_block_sizes(conn):

    print()
    print("=" * 70)
    print("BLOCK SIZE ANALYSIS")
    print("=" * 70)

    cursor = conn.cursor()

    query = """
        SELECT
            block_key,
            COUNT(*) AS block_size
        FROM blocks
        GROUP BY block_key
        ORDER BY block_size DESC
    """

    rows = cursor.execute(query).fetchall()

    if not rows:

        print("No blocks found.")
        return

    sizes = [
        row[1]
        for row in rows
    ]

    series = pd.Series(sizes)

    print(f"Total unique blocks : {len(sizes):,}")
    print(f"Minimum block size  : {series.min():,}")
    print(f"Maximum block size  : {series.max():,}")
    print(f"Median block size   : {series.median():.2f}")
    print(f"Average block size  : {series.mean():.2f}")
    print(
        f"95th percentile     : "
        f"{series.quantile(0.95):.2f}"
    )
    print(
        f"99th percentile     : "
        f"{series.quantile(0.99):.2f}"
    )

    large_blocks = [
        row
        for row in rows
        if row[1] > LARGE_BLOCK_LIMIT
    ]

    print()
    print(
        f"Blocks > {LARGE_BLOCK_LIMIT:,}: "
        f"{len(large_blocks):,}"
    )

    print()
    print("Top 20 largest blocks:")

    for block_key, size in rows[:20]:

        print(
            f"{size:>12,}  {block_key}"
        )


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(path):

    if not os.path.exists(path):

        print()
        print(
            "WARNING: Ground truth file not found:"
        )
        print(path)

        return {}

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    required = [
        "source1_entity_id",
        "matched_entity_ids"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        print(
            "WARNING: Ground truth columns missing:"
        )
        print(missing)

        return {}

    ground_truth = {}

    for row in df.itertuples(index=False):

        s1_id = safe_string(
            getattr(
                row,
                "source1_entity_id"
            )
        )

        matches = safe_string(
            getattr(
                row,
                "matched_entity_ids"
            )
        )

        if not s1_id:
            continue

        if matches:

            match_ids = {
                x.strip()
                for x in matches.split(",")
                if x.strip()
            }

        else:

            match_ids = set()

        ground_truth[s1_id] = match_ids

    print()
    print(
        f"Loaded ground truth: "
        f"{len(ground_truth):,} Source-1 records"
    )

    return ground_truth


# ============================================================
# CANDIDATE LOOKUP
# ============================================================

def get_candidate_entity_ids(
    conn,
    block_keys
):
    """
    Retrieve candidate records from SQLite.

    DISTINCT is unnecessary because the blocks table already
    guarantees unique (block_key, rec_id) pairs.

    We still use a Python set because a record can match
    multiple different blocks.
    """

    if not block_keys:
        return set()

    cursor = conn.cursor()

    candidates = set()

    for block_key in block_keys:

        rows = cursor.execute(
            """
            SELECT rec_id
            FROM blocks
            WHERE block_key = ?
            """,
            (block_key,)
        ).fetchall()

        for row in rows:
            candidates.add(row[0])

    return candidates


def get_entity_ids_from_rec_ids(
    conn,
    rec_ids
):

    if not rec_ids:
        return set()

    cursor = conn.cursor()

    entity_ids = set()

    # SQLite has a parameter limit, so query in batches.
    rec_ids = list(rec_ids)

    batch_size = 500

    for start in range(
        0,
        len(rec_ids),
        batch_size
    ):

        batch = rec_ids[
            start:start + batch_size
        ]

        placeholders = ",".join(
            "?"
            for _ in batch
        )

        query = f"""
            SELECT entity_id
            FROM records
            WHERE rec_id IN ({placeholders})
        """

        rows = cursor.execute(
            query,
            batch
        ).fetchall()

        for row in rows:

            if row[0]:
                entity_ids.add(
                    row[0]
                )

    return entity_ids


# ============================================================
# TRAIN CANDIDATE GENERATION + RECALL
# ============================================================

def generate_train_candidates(
    conn,
    source1_path,
    ground_truth,
    output_path
):

    print()
    print("=" * 70)
    print("GENERATING TRAIN CANDIDATES")
    print("=" * 70)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    total_s1 = 0
    zero_candidates = 0
    total_candidates = 0

    true_s2_total = 0
    true_s2_found = 0

    true_s3_total = 0
    true_s3_found = 0

    missed_records = []

    output_rows = []

    for chunk in pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    ):

        for row in chunk.itertuples(index=False):

            s1_entity_id = safe_string(
                getattr(
                    row,
                    "entity_id",
                    ""
                )
            )

            business_name = safe_string(
                getattr(
                    row,
                    "business_name",
                    ""
                )
            )

            business_address = safe_string(
                getattr(
                    row,
                    "business_address",
                    ""
                )
            )

            country = safe_string(
                getattr(
                    row,
                    "country",
                    ""
                )
            )

            block_keys = generate_block_keys(
                business_name,
                business_address,
                country
            )

            candidate_rec_ids = (
                get_candidate_entity_ids(
                    conn,
                    block_keys
                )
            )

            candidate_entity_ids = (
                get_entity_ids_from_rec_ids(
                    conn,
                    candidate_rec_ids
                )
            )

            total_s1 += 1

            candidate_count = len(
                candidate_entity_ids
            )

            total_candidates += candidate_count

            if candidate_count == 0:

                zero_candidates += 1

            # ------------------------------------------------
            # Ground truth evaluation
            # ------------------------------------------------

            true_matches = ground_truth.get(
                s1_entity_id,
                set()
            )

            true_s2 = {
                x
                for x in true_matches
                if x.startswith("S2-")
            }

            true_s3 = {
                x
                for x in true_matches
                if x.startswith("S3-")
            }

            found_s2 = (
                true_s2
                & candidate_entity_ids
            )

            found_s3 = (
                true_s3
                & candidate_entity_ids
            )

            true_s2_total += len(
                true_s2
            )

            true_s2_found += len(
                found_s2
            )

            true_s3_total += len(
                true_s3
            )

            true_s3_found += len(
                found_s3
            )

            missed = (
                true_matches
                - candidate_entity_ids
            )

            if missed:

                missed_records.append(
                    (
                        s1_entity_id,
                        missed
                    )
                )

            output_rows.append(
                {
                    "source1_entity_id":
                        s1_entity_id,

                    "candidate_entity_ids":
                        "|".join(
                            sorted(
                                candidate_entity_ids
                            )
                        ),

                    "candidate_count":
                        candidate_count
                }
            )

        # Write output incrementally
        if output_rows:

            write_header = not os.path.exists(
                output_path
            )

            output_df = pd.DataFrame(
                output_rows
            )

            output_df.to_csv(
                output_path,
                sep="\t",
                index=False,
                mode="w" if write_header else "a",
                header=write_header
            )

            output_rows.clear()

        print(
            f"Processed Source 1: "
            f"{total_s1:,} records"
        )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAIN BLOCKING RESULTS")
    print("=" * 70)

    print(
        f"Source 1 records       : "
        f"{total_s1:,}"
    )

    print(
        f"Zero-candidate records: "
        f"{zero_candidates:,}"
    )

    if total_s1:

        print(
            f"Zero-candidate rate   : "
            f"{zero_candidates / total_s1:.6f}"
        )

        print(
            f"Average candidates    : "
            f"{total_candidates / total_s1:.2f}"
        )

    if true_s2_total:

        s2_recall = (
            true_s2_found /
            true_s2_total
        )

        print(
            f"S2 blocking recall    : "
            f"{s2_recall:.6f}"
        )

    else:

        print(
            "S2 blocking recall    : "
            "N/A"
        )

    if true_s3_total:

        s3_recall = (
            true_s3_found /
            true_s3_total
        )

        print(
            f"S3 blocking recall    : "
            f"{s3_recall:.6f}"
        )

    else:

        print(
            "S3 blocking recall    : "
            "N/A"
        )

    total_true = (
        true_s2_total +
        true_s3_total
    )

    total_found = (
        true_s2_found +
        true_s3_found
    )

    if total_true:

        overall_recall = (
            total_found /
            total_true
        )

        print(
            f"Overall blocking recall: "
            f"{overall_recall:.6f}"
        )

    else:

        print(
            "Overall blocking recall: "
            "N/A"
        )

    print(
        f"Records with missed true matches: "
        f"{len(missed_records):,}"
    )

    if missed_records:

        print()
        print("First 20 missed records:")

        for s1_id, missed in missed_records[:20]:

            print(
                f"{s1_id} -> "
                f"{sorted(missed)}"
            )


# ============================================================
# TEST CANDIDATE GENERATION
# ============================================================

def generate_test_candidates(
    conn,
    source1_path,
    output_path
):

    print()
    print("=" * 70)
    print("GENERATING TEST CANDIDATES")
    print("=" * 70)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    if os.path.exists(output_path):
        os.remove(output_path)

    total_s1 = 0
    zero_candidates = 0
    total_candidates = 0

    for chunk in pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    ):

        output_rows = []

        for row in chunk.itertuples(index=False):

            s1_entity_id = safe_string(
                getattr(
                    row,
                    "entity_id",
                    ""
                )
            )

            business_name = safe_string(
                getattr(
                    row,
                    "business_name",
                    ""
                )
            )

            business_address = safe_string(
                getattr(
                    row,
                    "business_address",
                    ""
                )
            )

            country = safe_string(
                getattr(
                    row,
                    "country",
                    ""
                )
            )

            block_keys = generate_block_keys(
                business_name,
                business_address,
                country
            )

            candidate_rec_ids = (
                get_candidate_entity_ids(
                    conn,
                    block_keys
                )
            )

            candidate_entity_ids = (
                get_entity_ids_from_rec_ids(
                    conn,
                    candidate_rec_ids
                )
            )

            candidate_count = len(
                candidate_entity_ids
            )

            total_s1 += 1
            total_candidates += candidate_count

            if candidate_count == 0:
                zero_candidates += 1

            output_rows.append(
                {
                    "source1_entity_id":
                        s1_entity_id,

                    "candidate_entity_ids":
                        "|".join(
                            sorted(
                                candidate_entity_ids
                            )
                        ),

                    "candidate_count":
                        candidate_count
                }
            )

        if output_rows:

            output_df = pd.DataFrame(
                output_rows
            )

            write_header = (
                not os.path.exists(
                    output_path
                )
            )

            output_df.to_csv(
                output_path,
                sep="\t",
                index=False,
                mode="a",
                header=write_header
            )

        print(
            f"Processed TEST Source 1: "
            f"{total_s1:,}"
        )

    print()
    print("=" * 70)
    print("TEST CANDIDATE RESULTS")
    print("=" * 70)

    print(
        f"Source 1 records       : "
        f"{total_s1:,}"
    )

    print(
        f"Zero-candidate records: "
        f"{zero_candidates:,}"
    )

    if total_s1:

        print(
            f"Zero-candidate rate   : "
            f"{zero_candidates / total_s1:.6f}"
        )

        print(
            f"Average candidates    : "
            f"{total_candidates / total_s1:.2f}"
        )


# ============================================================
# DATABASE STATISTICS
# ============================================================

def database_statistics(
    conn
):

    cursor = conn.cursor()

    record_count = cursor.execute(
        "SELECT COUNT(*) FROM records"
    ).fetchone()[0]

    block_count = cursor.execute(
        "SELECT COUNT(*) FROM blocks"
    ).fetchone()[0]

    unique_blocks = cursor.execute(
        """
        SELECT COUNT(DISTINCT block_key)
        FROM blocks
        """
    ).fetchone()[0]

    print()
    print("=" * 70)
    print("DATABASE STATISTICS")
    print("=" * 70)

    print(
        f"Records             : "
        f"{record_count:,}"
    )

    print(
        f"Block assignments   : "
        f"{block_count:,}"
    )

    print(
        f"Unique block keys   : "
        f"{unique_blocks:,}"
    )

    if record_count:

        print(
            f"Blocks per record   : "
            f"{block_count / record_count:.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def run_train():

    print()
    print("#" * 70)
    print("TRAIN PHASE")
    print("#" * 70)

    conn = create_database(
        TRAIN_DB
    )

    next_rec_id = 1

    # IMPORTANT:
    # The same next_rec_id counter is passed from S2 to S3.
    # Therefore there are NO collisions.

    next_rec_id = index_source_file(
        conn,
        TRAIN_SOURCE2,
        "S2",
        next_rec_id
    )

    next_rec_id = index_source_file(
        conn,
        TRAIN_SOURCE3,
        "S3",
        next_rec_id
    )

    database_statistics(
        conn
    )

    analyze_block_sizes(
        conn
    )

    ground_truth = load_ground_truth(
        GROUND_TRUTH
    )

    if os.path.exists(TRAIN_OUTPUT):
        os.remove(TRAIN_OUTPUT)

    generate_train_candidates(
        conn,
        TRAIN_SOURCE1,
        ground_truth,
        TRAIN_OUTPUT
    )

    conn.close()

    print()
    print(
        f"TRAIN candidate file:"
    )
    print(
        TRAIN_OUTPUT
    )

    print()
    print(
        f"TRAIN database:"
    )
    print(
        TRAIN_DB
    )


def run_test():

    print()
    print("#" * 70)
    print("TEST PHASE")
    print("#" * 70)

    conn = create_database(
        TEST_DB
    )

    next_rec_id = 1

    next_rec_id = index_source_file(
        conn,
        TEST_SOURCE2,
        "S2",
        next_rec_id
    )

    next_rec_id = index_source_file(
        conn,
        TEST_SOURCE3,
        "S3",
        next_rec_id
    )

    database_statistics(
        conn
    )

    analyze_block_sizes(
        conn
    )

    generate_test_candidates(
        conn,
        TEST_SOURCE1,
        TEST_OUTPUT
    )

    conn.close()

    print()
    print(
        "TEST candidate file:"
    )
    print(
        TEST_OUTPUT
    )

    print()
    print(
        "TEST database:"
    )
    print(
        TEST_DB
    )


if __name__ == "__main__":

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("ENTITY RESOLUTION - BLOCKING PIPELINE")
    print("=" * 70)

    print()
    print("1. TRAIN")
    print("2. TEST")
    print("3. BOTH")

    choice = input(
        "\nEnter choice: "
    ).strip()

    if choice == "1":

        run_train()

    elif choice == "2":

        run_test()

    elif choice == "3":

        run_train()
        run_test()

    else:

        print(
            "Invalid choice."
        )
