import os
import re
import sqlite3
import unicodedata
import pandas as pd
import time
from collections import defaultdict, Counter
import hashlib

# ==========================================
# 1. ENHANCED NORMALIZATION
# ==========================================

ABBREVIATIONS = {
    'pvt': 'private', 'ltd': 'limited', 'llc': 'limited liability company',
    'corp': 'corporation', 'co': 'company', 'bros': 'brothers', 'est': 'establishment',
    'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard', 'rd': 'road',
    'dr': 'drive', 'ct': 'court', 'pl': 'place', 'hwy': 'highway',
    'ste': 'suite', 'fl': 'floor', 'dept': 'department', 'apt': 'apartment',
    'intl': 'international', 'mfg': 'manufacturing', 'dist': 'distributors'
}

STOPWORDS = {'the', 'and', 'or', 'of', 'in', 'at', 'near', 'by', 'for'}


def normalize_text(text):
    """Enhanced normalization with better symbol handling."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = unicodedata.normalize('NFKC', text)
    text = text.casefold()

    # Replace common symbols BEFORE removing punctuation
    text = text.replace('&', ' and ')
    text = text.replace('@', ' at ')

    # Remove punctuation but keep Unicode letters and digits
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()

    # Expand abbreviations (ASCII only to preserve Indic scripts)
    words = text.split()
    expanded = [ABBREVIATIONS.get(w, w) if w.isascii() and w.isalpha() else w for w in words]

    return ' '.join(expanded)


def normalize_country(country):
    return normalize_text(country)


def extract_address_components(addr):
    """Enhanced address parsing with city/state extraction."""
    if not isinstance(addr, str) or not addr.strip():
        return "", "", "", "", ""

    # Postcode
    pc_match = re.search(r'\b(?:\d{5,6}|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b', addr, re.IGNORECASE)
    postcode = pc_match.group(0).replace(" ", "") if pc_match else ""

    # House number
    hn_match = re.search(r'\b\d{1,4}\b', addr)
    house_num = hn_match.group(0) if hn_match else ""
    if house_num == postcode:
        house_num = ""

    # Remove extracted numbers
    street = addr
    if house_num:
        street = street.replace(house_num, "", 1)
    if postcode:
        street = street.replace(postcode, "", 1)
    street = re.sub(r'\s+', ' ', street).strip()

    # Try to extract city/state (heuristic: last 1-2 words if they look like proper nouns)
    words = street.split()
    city = ""
    state = ""
    if len(words) >= 2:
        # Simple heuristic: assume last word is state, second-to-last is city
        if words[-1].isalpha() and len(words[-1]) > 2:
            state = words[-1]
            if len(words) >= 3 and words[-2].isalpha() and len(words[-2]) > 2:
                city = words[-2]

    return postcode, house_num, street, city, state


def simple_soundex(word):
    """Simple Soundex for phonetic blocking."""
    if not word or not word[0].isalpha():
        return ""

    word = word.upper()
    soundex = word[0]

    mapping = {
        'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3', 'L': '4',
        'MN': '5', 'R': '6'
    }

    prev_code = ''
    for char in word[1:]:
        code = ''
        for chars, num in mapping.items():
            if char in chars:
                code = num
                break

        if code and code != prev_code:
            soundex += code
            if len(soundex) == 4:
                break

        if code:
            prev_code = code

    return soundex.ljust(4, '0')


def get_rare_tokens(text, min_length=4):
    """Extract tokens that are likely rare/unique."""
    tokens = text.split()
    # For now, just return tokens >= min_length (we'll filter by frequency later)
    return [t for t in tokens if len(t) >= min_length and t not in STOPWORDS]


def generate_block_keys(norm_name, norm_addr, norm_country, entity_id):
    """Generate comprehensive high-recall blocking keys."""
    keys = []
    if not norm_name or not norm_country:
        return keys

    tokens = norm_name.split()
    sorted_tokens = ' '.join(sorted(tokens))

    # === NAME-BASED BLOCKS ===
    # 1. Exact name + country
    keys.append(f"cn:{norm_country}:{norm_name}")

    # 2. Sorted tokens (handles transpositions)
    keys.append(f"cns:{norm_country}:{sorted_tokens}")

    # 3. First 2 tokens
    if len(tokens) >= 2:
        keys.append(f"cnt:{norm_country}:{' '.join(tokens[:2])}")

    # 4. Last 2 tokens (catches "XYZ Pvt Ltd" vs "Pvt Ltd XYZ")
    if len(tokens) >= 2:
        keys.append(f"cnt_last:{norm_country}:{' '.join(tokens[-2:])}")

    # 5. Individual token blocking (each word becomes a key)
    for i, token in enumerate(tokens):
        if len(token) >= 3:
            keys.append(f"tok:{norm_country}:{token}:{i}")  # position-aware

    # 6. Phonetic blocking (Soundex on first 2 tokens)
    if len(tokens) >= 1:
        soundex1 = simple_soundex(tokens[0])
        if soundex1:
            keys.append(f"phon:{norm_country}:{soundex1}")
    if len(tokens) >= 2:
        soundex2 = simple_soundex(tokens[1])
        if soundex2:
            keys.append(f"phon2:{norm_country}:{soundex2}")

    # 7. Bigram blocking (character pairs)
    if len(norm_name) >= 4:
        bigrams = [norm_name[i:i + 2] for i in range(len(norm_name) - 1)]
        # Use first 3 bigrams as a key
        if len(bigrams) >= 3:
            keys.append(f"bigram:{norm_country}:{'|'.join(bigrams[:3])}")

    # === ADDRESS-BASED BLOCKS ===
    if norm_addr:
        postcode, house_num, street, city, state = extract_address_components(norm_addr)

        # 8. Postcode + name
        if postcode:
            keys.append(f"pn:{postcode}:{norm_name}")

        # 9. House number + name
        if house_num:
            keys.append(f"hn:{house_num}:{norm_name}")

        # 10. Street + name
        if street and len(street) > 3:
            keys.append(f"sn:{street}:{norm_name}")

        # 11. City + name
        if city:
            keys.append(f"city:{city}:{norm_name}")

        # 12. State + name
        if state:
            keys.append(f"state:{state}:{norm_name}")

        # 13. Postcode + first token of name
        if postcode and tokens:
            keys.append(f"pn_tok:{postcode}:{tokens[0]}")

        # 14. House number + first 2 tokens
        if house_num and len(tokens) >= 2:
            keys.append(f"hn_tok:{house_num}:{' '.join(tokens[:2])}")

    return keys


# ==========================================
# 2. DATABASE SETUP
# ==========================================

def setup_db(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA cache_size = 300000")

    conn.execute("""
        CREATE TABLE records (
            rec_id INTEGER PRIMARY KEY,
            entity_id TEXT,
            source TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE blocks (
            block_key TEXT,
            rec_id INTEGER
        )
    """)

    conn.execute("CREATE INDEX idx_blocks_key ON blocks(block_key)")
    conn.execute("CREATE INDEX idx_blocks_rec ON blocks(rec_id)")
    conn.execute("CREATE INDEX idx_records_entity ON records(entity_id)")
    conn.execute("CREATE INDEX idx_records_source ON records(source)")

    conn.commit()
    return conn


# ==========================================
# 3. INDEXING WITH BLOCK SIZE TRACKING
# ==========================================

def index_source_files(conn, file_paths, source_name):
    print(f"  -> Indexing {source_name}...")
    rec_id_counter = [0]
    block_sizes = Counter()

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"     ⚠️  File not found: {file_path}")
            continue

        start_time = time.time()
        chunk_count = 0

        for chunk in pd.read_csv(file_path, sep='\t', chunksize=50000, dtype=str):
            chunk_count += 1
            chunk = chunk.fillna("")
            records_batch = []
            blocks_batch = []

            for _, row in chunk.iterrows():
                rec_id = rec_id_counter[0]
                rec_id_counter[0] += 1
                records_batch.append((rec_id, row['entity_id'], source_name))

                norm_name = normalize_text(row['business_name'])
                norm_addr = normalize_text(row['business_address'])
                norm_country = normalize_country(row['country'])

                keys = generate_block_keys(norm_name, norm_addr, norm_country, row['entity_id'])
                for k in keys:
                    blocks_batch.append((k, rec_id))
                    block_sizes[k] += 1

            conn.execute("BEGIN")
            conn.executemany("INSERT INTO records VALUES (?, ?, ?)", records_batch)
            conn.executemany("INSERT INTO blocks VALUES (?, ?)", blocks_batch)
            conn.commit()

        elapsed = time.time() - start_time
        print(f"     Processed {chunk_count} chunks in {elapsed:.1f}s")

    # Report large blocks
    large_blocks = [(k, v) for k, v in block_sizes.items() if v > 10000]
    if large_blocks:
        print(f"     ⚠️  Found {len(large_blocks)} large blocks (>10k records)")
        for k, v in sorted(large_blocks, key=lambda x: -x[1])[:5]:
            print(f"        {k}: {v:,} records")

    return block_sizes


# ==========================================
# 4. CANDIDATE GENERATION WITH DIAGNOSTICS
# ==========================================

def load_ground_truth(gt_path):
    gt = {}
    if not os.path.exists(gt_path):
        return gt
    df = pd.read_csv(gt_path, sep='\t', dtype=str)
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).strip()
        if matches and matches != 'nan':
            gt[s1_id] = set(matches.split(','))
        else:
            gt[s1_id] = set()
    return gt


def analyze_missed_matches(conn, gt_dict, s1_id, true_matches, candidates):
    """Analyze why a true match was missed."""
    missed = true_matches - set(candidates)
    if not missed:
        return []

    analysis = []
    for missed_id in missed:
        # Get the record details
        cursor = conn.execute(
            "SELECT entity_id, source FROM records WHERE entity_id = ?",
            (missed_id,)
        )
        row = cursor.fetchone()
        if not row:
            analysis.append(f"{missed_id}: NOT IN DATABASE")
            continue

        # Get blocking keys for both records
        s1_cursor = conn.execute(
            "SELECT block_key FROM blocks WHERE rec_id IN (SELECT rec_id FROM records WHERE entity_id = ?)",
            (s1_id,)
        )
        s1_keys = set(r[0] for r in s1_cursor.fetchall())

        missed_cursor = conn.execute(
            "SELECT block_key FROM blocks WHERE rec_id IN (SELECT rec_id FROM records WHERE entity_id = ?)",
            (missed_id,)
        )
        missed_keys = set(r[0] for r in missed_cursor.fetchall())

        overlap = s1_keys & missed_keys
        if overlap:
            analysis.append(f"{missed_id} ({row[1]}): KEYS OVERLAP BUT NOT IN CANDIDATES (BUG)")
        else:
            analysis.append(f"{missed_id} ({row[1]}): NO KEY OVERLAP - blocking keys don't match")

    return analysis


def generate_candidates_with_diagnostics(conn, s1_files, output_tsv, gt_dict=None, source_filter=None):
    print(f"  -> Generating candidates for S1...")

    total_s1 = 0
    total_candidates = 0
    zero_candidate_s1 = 0

    # Per-source tracking
    s2_recall_covered = 0
    s2_recall_total = 0
    s3_recall_covered = 0
    s3_recall_total = 0

    # Per-key recall tracking
    key_recall = defaultdict(lambda: {'covered': 0, 'total': 0})

    missed_matches_analysis = []

    with open(output_tsv, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

        for file_path in s1_files:
            if not os.path.exists(file_path):
                continue

            for chunk in pd.read_csv(file_path, sep='\t', chunksize=50000, dtype=str):
                chunk = chunk.fillna("")

                for _, row in chunk.iterrows():
                    s1_id = row['entity_id']
                    total_s1 += 1

                    norm_name = normalize_text(row['business_name'])
                    norm_addr = normalize_text(row['business_address'])
                    norm_country = normalize_country(row['country'])
                    keys = generate_block_keys(norm_name, norm_addr, norm_country, s1_id)

                    # Query candidates
                    candidates = []
                    if keys:
                        placeholders = ','.join(['?'] * len(keys))
                        query = f"""
                            SELECT DISTINCT r.entity_id, r.source
                            FROM blocks b
                            JOIN records r ON b.rec_id = r.rec_id
                            WHERE b.block_key IN ({placeholders})
                        """
                        if source_filter:
                            query += f" AND r.source IN ({','.join(['?'] * len(source_filter))})"
                            cursor = conn.execute(query, keys + list(source_filter))
                        else:
                            cursor = conn.execute(query, keys)

                        candidates = [r[0] for r in cursor.fetchall()]

                    # Write to TSV
                    f.write(f"{s1_id}\t{','.join(candidates)}\n")

                    if not candidates:
                        zero_candidate_s1 += 1

                    total_candidates += len(candidates)

                    # Evaluate recall if GT is provided
                    if gt_dict and s1_id in gt_dict:
                        true_matches = gt_dict[s1_id]

                        # Separate S2 and S3 matches
                        s2_matches = {m for m in true_matches if m.startswith('S2-')}
                        s3_matches = {m for m in true_matches if m.startswith('S3-')}

                        if s2_matches:
                            s2_recall_total += len(s2_matches)
                            s2_covered = len(s2_matches.intersection(set(candidates)))
                            s2_recall_covered += s2_covered

                        if s3_matches:
                            s3_recall_total += len(s3_matches)
                            s3_covered = len(s3_matches.intersection(set(candidates)))
                            s3_recall_covered += s3_covered

                        # Analyze missed matches
                        if true_matches:
                            missed = true_matches - set(candidates)
                            if missed:
                                analysis = analyze_missed_matches(conn, gt_dict, s1_id, true_matches, candidates)
                                missed_matches_analysis.extend(analysis)

    # Print diagnostics
    print(f"\n{'=' * 60}")
    print(f"BLOCKING DIAGNOSTICS")
    print(f"{'=' * 60}")
    print(f"Total S1 records: {total_s1:,}")
    print(f"S1 records with zero candidates: {zero_candidate_s1:,} ({zero_candidate_s1 / total_s1 * 100:.2f}%)")
    print(f"Total candidates generated: {total_candidates:,}")
    print(f"Average candidates per S1: {total_candidates / total_s1:.1f}")

    if gt_dict:
        print(f"\n--- RECALL METRICS ---")

        # Overall recall
        total_true = s2_recall_total + s3_recall_total
        total_covered = s2_recall_covered + s3_recall_covered
        overall_recall = (total_covered / total_true * 100) if total_true > 0 else 0
        print(f"Overall blocking recall: {overall_recall:.2f}% ({total_covered}/{total_true})")

        # S1→S2 recall
        s2_recall = (s2_recall_covered / s2_recall_total * 100) if s2_recall_total > 0 else 0
        print(f"S1→S2 recall: {s2_recall:.2f}% ({s2_recall_covered}/{s2_recall_total})")

        # S1→S3 recall
        s3_recall = (s3_recall_covered / s3_recall_total * 100) if s3_recall_total > 0 else 0
        print(f"S1→S3 recall: {s3_recall:.2f}% ({s3_recall_covered}/{s3_recall_total})")

        # Missed matches
        total_missed = total_true - total_covered
        print(f"True matches missed: {total_missed:,}")

        if missed_matches_analysis:
            print(f"\n--- SAMPLE MISSED MATCHES (first 10) ---")
            for analysis in missed_matches_analysis[:10]:
                print(f"  {analysis}")

    print(f"{'=' * 60}\n")

    return {
        'total_s1': total_s1,
        'zero_candidates': zero_candidate_s1,
        'total_candidates': total_candidates,
        'overall_recall': overall_recall if gt_dict else None,
        's2_recall': s2_recall if gt_dict else None,
        's3_recall': s3_recall if gt_dict else None
    }


# ==========================================
# 5. BLOCK SIZE ANALYSIS
# ==========================================

def analyze_block_sizes(conn):
    print("\n--- BLOCK SIZE ANALYSIS ---")

    # Get block size distribution
    cursor = conn.execute("""
        SELECT block_key, COUNT(*) as cnt
        FROM blocks
        GROUP BY block_key
        ORDER BY cnt DESC
    """)

    block_sizes = [row[1] for row in cursor.fetchall()]

    if not block_sizes:
        print("No blocks found!")
        return

    print(f"Total unique blocks: {len(block_sizes):,}")
    print(f"Min block size: {min(block_sizes):,}")
    print(f"Max block size: {max(block_sizes):,}")
    print(f"Median block size: {sorted(block_sizes)[len(block_sizes) // 2]:,}")
    print(f"Average block size: {sum(block_sizes) / len(block_sizes):.1f}")

    # Percentiles
    sorted_sizes = sorted(block_sizes)
    n = len(sorted_sizes)
    print(f"\nBlock size percentiles:")
    print(f"  50th: {sorted_sizes[int(n * 0.5)]:,}")
    print(f"  90th: {sorted_sizes[int(n * 0.9)]:,}")
    print(f"  95th: {sorted_sizes[int(n * 0.95)]:,}")
    print(f"  99th: {sorted_sizes[int(n * 0.99)]:,}")

    # Top 10 largest blocks
    cursor = conn.execute("""
        SELECT block_key, COUNT(*) as cnt
        FROM blocks
        GROUP BY block_key
        ORDER BY cnt DESC
        LIMIT 10
    """)

    print(f"\nTop 10 largest blocks:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,} records")


# ==========================================
# 6. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    import os
    import re
    import sqlite3
    import unicodedata
    import pandas as pd
    import time
    from collections import defaultdict, Counter

    # ==========================================
    # 1. ENHANCED NORMALIZATION
    # ==========================================

    ABBREVIATIONS = {
        'pvt': 'private', 'ltd': 'limited', 'llc': 'limited liability company',
        'corp': 'corporation', 'co': 'company', 'bros': 'brothers', 'est': 'establishment',
        'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard', 'rd': 'road',
        'dr': 'drive', 'ct': 'court', 'pl': 'place', 'hwy': 'highway',
        'ste': 'suite', 'fl': 'floor', 'dept': 'department', 'apt': 'apartment',
        'intl': 'international', 'mfg': 'manufacturing', 'dist': 'distributors'
    }

    STOPWORDS = {'the', 'and', 'or', 'of', 'in', 'at', 'near', 'by', 'for'}


    def normalize_text(text):
        """Enhanced normalization with better symbol handling."""
        if not isinstance(text, str) or not text.strip():
            return ""

        text = unicodedata.normalize('NFKC', text)
        text = text.casefold()

        # Replace common symbols BEFORE removing punctuation
        text = text.replace('&', ' and ')
        text = text.replace('@', ' at ')

        # Remove punctuation but keep Unicode letters and digits
        text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
        text = re.sub(r'\s+', ' ', text).strip()

        # Expand abbreviations (ASCII only to preserve Indic scripts)
        words = text.split()
        expanded = [ABBREVIATIONS.get(w, w) if w.isascii() and w.isalpha() else w for w in words]

        return ' '.join(expanded)


    def normalize_country(country):
        return normalize_text(country)


    def extract_address_components(addr):
        """Enhanced address parsing with city/state extraction."""
        if not isinstance(addr, str) or not addr.strip():
            return "", "", "", "", ""

        # Postcode
        pc_match = re.search(r'\b(?:\d{5,6}|[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b', addr, re.IGNORECASE)
        postcode = pc_match.group(0).replace(" ", "") if pc_match else ""

        # House number
        hn_match = re.search(r'\b\d{1,4}\b', addr)
        house_num = hn_match.group(0) if hn_match else ""
        if house_num == postcode:
            house_num = ""

        # Remove extracted numbers
        street = addr
        if house_num:
            street = street.replace(house_num, "", 1)
        if postcode:
            street = street.replace(postcode, "", 1)
        street = re.sub(r'\s+', ' ', street).strip()

        # Try to extract city/state (heuristic: last 1-2 words if they look like proper nouns)
        words = street.split()
        city = ""
        state = ""
        if len(words) >= 2:
            # Simple heuristic: assume last word is state, second-to-last is city
            if words[-1].isalpha() and len(words[-1]) > 2:
                state = words[-1]
                if len(words) >= 3 and words[-2].isalpha() and len(words[-2]) > 2:
                    city = words[-2]

        return postcode, house_num, street, city, state


    def simple_soundex(word):
        """Simple Soundex for phonetic blocking."""
        if not word or not word[0].isalpha():
            return ""

        word = word.upper()
        soundex = word[0]

        mapping = {
            'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3', 'L': '4',
            'MN': '5', 'R': '6'
        }

        prev_code = ''
        for char in word[1:]:
            code = ''
            for chars, num in mapping.items():
                if char in chars:
                    code = num
                    break

            if code and code != prev_code:
                soundex += code
                if len(soundex) == 4:
                    break

            if code:
                prev_code = code

        return soundex.ljust(4, '0')


    def generate_block_keys(norm_name, norm_addr, norm_country, entity_id):
        """Generate comprehensive high-recall blocking keys."""
        keys = []
        if not norm_name or not norm_country:
            return keys

        tokens = norm_name.split()
        sorted_tokens = ' '.join(sorted(tokens))

        # === NAME-BASED BLOCKS ===
        # 1. Exact name + country
        keys.append(f"cn:{norm_country}:{norm_name}")

        # 2. Sorted tokens (handles transpositions)
        keys.append(f"cns:{norm_country}:{sorted_tokens}")

        # 3. First 2 tokens
        if len(tokens) >= 2:
            keys.append(f"cnt:{norm_country}:{' '.join(tokens[:2])}")

        # 4. Last 2 tokens (catches "XYZ Pvt Ltd" vs "Pvt Ltd XYZ")
        if len(tokens) >= 2:
            keys.append(f"cnt_last:{norm_country}:{' '.join(tokens[-2:])}")

        # 5. Individual token blocking (each word becomes a key)
        for i, token in enumerate(tokens):
            if len(token) >= 3:
                keys.append(f"tok:{norm_country}:{token}:{i}")  # position-aware

        # 6. Phonetic blocking (Soundex on first 2 tokens)
        if len(tokens) >= 1:
            soundex1 = simple_soundex(tokens[0])
            if soundex1:
                keys.append(f"phon:{norm_country}:{soundex1}")
        if len(tokens) >= 2:
            soundex2 = simple_soundex(tokens[1])
            if soundex2:
                keys.append(f"phon2:{norm_country}:{soundex2}")

        # 7. Bigram blocking (character pairs)
        if len(norm_name) >= 4:
            bigrams = [norm_name[i:i + 2] for i in range(len(norm_name) - 1)]
            # Use first 3 bigrams as a key
            if len(bigrams) >= 3:
                keys.append(f"bigram:{norm_country}:{'|'.join(bigrams[:3])}")

        # === ADDRESS-BASED BLOCKS ===
        if norm_addr:
            postcode, house_num, street, city, state = extract_address_components(norm_addr)

            # 8. Postcode + name
            if postcode:
                keys.append(f"pn:{postcode}:{norm_name}")

            # 9. House number + name
            if house_num:
                keys.append(f"hn:{house_num}:{norm_name}")

            # 10. Street + name
            if street and len(street) > 3:
                keys.append(f"sn:{street}:{norm_name}")

            # 11. City + name
            if city:
                keys.append(f"city:{city}:{norm_name}")

            # 12. State + name
            if state:
                keys.append(f"state:{state}:{norm_name}")

            # 13. Postcode + first token of name
            if postcode and tokens:
                keys.append(f"pn_tok:{postcode}:{tokens[0]}")

            # 14. House number + first 2 tokens
            if house_num and len(tokens) >= 2:
                keys.append(f"hn_tok:{house_num}:{' '.join(tokens[:2])}")

        return keys


    # ==========================================
    # 2. DATABASE SETUP
    # ==========================================

    def setup_db(db_path):
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA cache_size = 300000")

        conn.execute("""
            CREATE TABLE records (
                rec_id INTEGER PRIMARY KEY,
                entity_id TEXT,
                source TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE blocks (
                block_key TEXT,
                rec_id INTEGER
            )
        """)

        conn.execute("CREATE INDEX idx_blocks_key ON blocks(block_key)")
        conn.execute("CREATE INDEX idx_blocks_rec ON blocks(rec_id)")
        conn.execute("CREATE INDEX idx_records_entity ON records(entity_id)")
        conn.execute("CREATE INDEX idx_records_source ON records(source)")

        conn.commit()
        return conn


    # ==========================================
    # 3. INDEXING WITH BLOCK SIZE TRACKING
    # ==========================================

    def index_source_files(conn, file_paths, source_name):
        print(f"  -> Indexing {source_name}...")
        rec_id_counter = [0]
        block_sizes = Counter()

        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"     ⚠️  File not found: {file_path}")
                continue

            start_time = time.time()
            chunk_count = 0

            for chunk in pd.read_csv(file_path, sep='\t', chunksize=50000, dtype=str):
                chunk_count += 1
                chunk = chunk.fillna("")
                records_batch = []
                blocks_batch = []

                for _, row in chunk.iterrows():
                    rec_id = rec_id_counter[0]
                    rec_id_counter[0] += 1
                    records_batch.append((rec_id, row['entity_id'], source_name))

                    norm_name = normalize_text(row['business_name'])
                    norm_addr = normalize_text(row['business_address'])
                    norm_country = normalize_country(row['country'])

                    keys = generate_block_keys(norm_name, norm_addr, norm_country, row['entity_id'])
                    for k in keys:
                        blocks_batch.append((k, rec_id))
                        block_sizes[k] += 1

                conn.execute("BEGIN")
                conn.executemany("INSERT INTO records VALUES (?, ?, ?)", records_batch)
                conn.executemany("INSERT INTO blocks VALUES (?, ?)", blocks_batch)
                conn.commit()

            elapsed = time.time() - start_time
            print(f"     Processed {chunk_count} chunks in {elapsed:.1f}s")

        # Report large blocks
        large_blocks = [(k, v) for k, v in block_sizes.items() if v > 10000]
        if large_blocks:
            print(f"     ⚠️  Found {len(large_blocks)} large blocks (>10k records)")
            for k, v in sorted(large_blocks, key=lambda x: -x[1])[:5]:
                print(f"        {k}: {v:,} records")

        return block_sizes


    # ==========================================
    # 4. CANDIDATE GENERATION WITH DIAGNOSTICS
    # ==========================================

    def load_ground_truth(gt_path):
        gt = {}
        if not os.path.exists(gt_path):
            return gt
        df = pd.read_csv(gt_path, sep='\t', dtype=str)
        for _, row in df.iterrows():
            s1_id = row['source1_entity_id']
            matches = str(row['matched_entity_ids']).strip()
            if matches and matches != 'nan':
                gt[s1_id] = set(matches.split(','))
            else:
                gt[s1_id] = set()
        return gt


    def analyze_missed_matches(conn, gt_dict, s1_id, true_matches, candidates):
        """Analyze why a true match was missed."""
        missed = true_matches - set(candidates)
        if not missed:
            return []

        analysis = []
        for missed_id in missed:
            # Get the record details
            cursor = conn.execute(
                "SELECT entity_id, source FROM records WHERE entity_id = ?",
                (missed_id,)
            )
            row = cursor.fetchone()
            if not row:
                analysis.append(f"{missed_id}: NOT IN DATABASE")
                continue

            # Get blocking keys for both records
            s1_cursor = conn.execute(
                "SELECT block_key FROM blocks WHERE rec_id IN (SELECT rec_id FROM records WHERE entity_id = ?)",
                (s1_id,)
            )
            s1_keys = set(r[0] for r in s1_cursor.fetchall())

            missed_cursor = conn.execute(
                "SELECT block_key FROM blocks WHERE rec_id IN (SELECT rec_id FROM records WHERE entity_id = ?)",
                (missed_id,)
            )
            missed_keys = set(r[0] for r in missed_cursor.fetchall())

            overlap = s1_keys & missed_keys
            if overlap:
                analysis.append(f"{missed_id} ({row[1]}): KEYS OVERLAP BUT NOT IN CANDIDATES (BUG)")
            else:
                analysis.append(f"{missed_id} ({row[1]}): NO KEY OVERLAP - blocking keys don't match")

        return analysis


    def generate_candidates_with_diagnostics(conn, s1_files, output_tsv, gt_dict=None, source_filter=None):
        print(f"  -> Generating candidates for S1...")

        total_s1 = 0
        total_candidates = 0
        zero_candidate_s1 = 0

        # Per-source tracking
        s2_recall_covered = 0
        s2_recall_total = 0
        s3_recall_covered = 0
        s3_recall_total = 0

        missed_matches_analysis = []

        with open(output_tsv, 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")

            for file_path in s1_files:
                if not os.path.exists(file_path):
                    continue

                for chunk in pd.read_csv(file_path, sep='\t', chunksize=50000, dtype=str):
                    chunk = chunk.fillna("")

                    for _, row in chunk.iterrows():
                        s1_id = row['entity_id']
                        total_s1 += 1

                        norm_name = normalize_text(row['business_name'])
                        norm_addr = normalize_text(row['business_address'])
                        norm_country = normalize_country(row['country'])
                        keys = generate_block_keys(norm_name, norm_addr, norm_country, s1_id)

                        # Query candidates
                        candidates = []
                        if keys:
                            placeholders = ','.join(['?'] * len(keys))
                            query = f"""
                                SELECT DISTINCT r.entity_id, r.source
                                FROM blocks b
                                JOIN records r ON b.rec_id = r.rec_id
                                WHERE b.block_key IN ({placeholders})
                            """
                            if source_filter:
                                query += f" AND r.source IN ({','.join(['?'] * len(source_filter))})"
                                cursor = conn.execute(query, keys + list(source_filter))
                            else:
                                cursor = conn.execute(query, keys)

                            candidates = [r[0] for r in cursor.fetchall()]

                        # Write to TSV
                        f.write(f"{s1_id}\t{','.join(candidates)}\n")

                        if not candidates:
                            zero_candidate_s1 += 1

                        total_candidates += len(candidates)

                        # Evaluate recall if GT is provided
                        if gt_dict and s1_id in gt_dict:
                            true_matches = gt_dict[s1_id]

                            # Separate S2 and S3 matches
                            s2_matches = {m for m in true_matches if m.startswith('S2-')}
                            s3_matches = {m for m in true_matches if m.startswith('S3-')}

                            if s2_matches:
                                s2_recall_total += len(s2_matches)
                                s2_covered = len(s2_matches.intersection(set(candidates)))
                                s2_recall_covered += s2_covered

                            if s3_matches:
                                s3_recall_total += len(s3_matches)
                                s3_covered = len(s3_matches.intersection(set(candidates)))
                                s3_recall_covered += s3_covered

                            # Analyze missed matches
                            if true_matches:
                                missed = true_matches - set(candidates)
                                if missed:
                                    analysis = analyze_missed_matches(conn, gt_dict, s1_id, true_matches, candidates)
                                    missed_matches_analysis.extend(analysis)

        # Print diagnostics
        print(f"\n{'=' * 60}")
        print(f"BLOCKING DIAGNOSTICS")
        print(f"{'=' * 60}")
        print(f"Total S1 records: {total_s1:,}")
        print(f"S1 records with zero candidates: {zero_candidate_s1:,} ({zero_candidate_s1 / total_s1 * 100:.2f}%)")
        print(f"Total candidates generated: {total_candidates:,}")
        print(f"Average candidates per S1: {total_candidates / total_s1:.1f}")

        if gt_dict:
            print(f"\n--- RECALL METRICS ---")

            # Overall recall
            total_true = s2_recall_total + s3_recall_total
            total_covered = s2_recall_covered + s3_recall_covered
            overall_recall = (total_covered / total_true * 100) if total_true > 0 else 0
            print(f"Overall blocking recall: {overall_recall:.2f}% ({total_covered}/{total_true})")

            # S1→S2 recall
            s2_recall = (s2_recall_covered / s2_recall_total * 100) if s2_recall_total > 0 else 0
            print(f"S1→S2 recall: {s2_recall:.2f}% ({s2_recall_covered}/{s2_recall_total})")

            # S1→S3 recall
            s3_recall = (s3_recall_covered / s3_recall_total * 100) if s3_recall_total > 0 else 0
            print(f"S1→S3 recall: {s3_recall:.2f}% ({s3_recall_covered}/{s3_recall_total})")

            # Missed matches
            total_missed = total_true - total_covered
            print(f"True matches missed: {total_missed:,}")

            if missed_matches_analysis:
                print(f"\n--- SAMPLE MISSED MATCHES (first 10) ---")
                for analysis in missed_matches_analysis[:10]:
                    print(f"  {analysis}")

        print(f"{'=' * 60}\n")

        return {
            'total_s1': total_s1,
            'zero_candidates': zero_candidate_s1,
            'total_candidates': total_candidates,
            'overall_recall': overall_recall if gt_dict else None,
            's2_recall': s2_recall if gt_dict else None,
            's3_recall': s3_recall if gt_dict else None
        }


    # ==========================================
    # 5. BLOCK SIZE ANALYSIS
    # ==========================================

    def analyze_block_sizes(conn):
        print("\n--- BLOCK SIZE ANALYSIS ---")

        # Get block size distribution
        cursor = conn.execute("""
            SELECT block_key, COUNT(*) as cnt
            FROM blocks
            GROUP BY block_key
            ORDER BY cnt DESC
        """)

        block_sizes = [row[1] for row in cursor.fetchall()]

        if not block_sizes:
            print("No blocks found!")
            return

        print(f"Total unique blocks: {len(block_sizes):,}")
        print(f"Min block size: {min(block_sizes):,}")
        print(f"Max block size: {max(block_sizes):,}")
        print(f"Median block size: {sorted(block_sizes)[len(block_sizes) // 2]:,}")
        print(f"Average block size: {sum(block_sizes) / len(block_sizes):.1f}")

        # Percentiles
        sorted_sizes = sorted(block_sizes)
        n = len(sorted_sizes)
        print(f"\nBlock size percentiles:")
        print(f"  50th: {sorted_sizes[int(n * 0.5)]:,}")
        print(f"  90th: {sorted_sizes[int(n * 0.9)]:,}")
        print(f"  95th: {sorted_sizes[int(n * 0.95)]:,}")
        print(f"  99th: {sorted_sizes[int(n * 0.99)]:,}")

        # Top 10 largest blocks
        cursor = conn.execute("""
            SELECT block_key, COUNT(*) as cnt
            FROM blocks
            GROUP BY block_key
            ORDER BY cnt DESC
            LIMIT 10
        """)

        print(f"\nTop 10 largest blocks:")
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]:,} records")


    # ==========================================
    # 6. MAIN EXECUTION - TRAIN ONLY
    # ==========================================

    if __name__ == "__main__":
        TRAIN_DIR = "train"

        # --- TRAIN PHASE ONLY ---
        print("=" * 60)
        print("PHASE 1: TRAIN BLOCKING & DIAGNOSTICS")
        print("=" * 60)

        train_db = setup_db("train_blocking.db")

        # Index S2 and S3
        index_source_files(train_db, [os.path.join(TRAIN_DIR, "train_source2.tsv")], "S2")
        index_source_files(train_db, [os.path.join(TRAIN_DIR, "train_source3.tsv")], "S3")

        # Analyze block sizes
        analyze_block_sizes(train_db)

        # Load ground truth
        gt_dict = load_ground_truth(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))

        # Generate candidates with full diagnostics
        generate_candidates_with_diagnostics(
            train_db,
            [os.path.join(TRAIN_DIR, "train_source1.tsv")],
            "train_candidates.tsv",
            gt_dict
        )

        train_db.close()

        print("\n✅ TRAIN PHASE COMPLETE!")
        print("Check the diagnostics above to see if blocking recall is good.")
        print("Once you're satisfied, we'll add the TEST phase.")