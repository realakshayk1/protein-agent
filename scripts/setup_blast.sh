#!/usr/bin/env bash
# =============================================================================
# setup_blast.sh  —  One-time local BLAST+ + SwissProt database setup
#
# Run this once from the repo root before the first benchmark run.
# After completion, the benchmark will use local BLAST+ automatically
# (no NCBI network calls, ~2-5s per query vs 30-60s remote).
#
# Requirements:
#   - BLAST+ installed (blastp on PATH)
#       Ubuntu/Debian:  sudo apt install ncbi-blast+
#       macOS:          brew install blast
#
# Usage:
#   bash scripts/setup_blast.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DB_DIR="$REPO_ROOT/data/blast_db"

echo "=== Local BLAST+ Setup ==="
echo "DB directory: $DB_DIR"

# ------------------------------------------------------------------
# 1. Check blastp is available
# ------------------------------------------------------------------
if ! command -v blastp &>/dev/null; then
    echo ""
    echo "ERROR: blastp not found on PATH."
    echo "Install BLAST+:"
    echo "  Ubuntu/Debian:  sudo apt install ncbi-blast+"
    echo "  macOS:          brew install blast"
    exit 1
fi
echo "blastp found: $(command -v blastp)"
echo "version: $(blastp -version 2>&1 | head -1)"

# ------------------------------------------------------------------
# 2. Create DB directory
# ------------------------------------------------------------------
mkdir -p "$DB_DIR"
cd "$DB_DIR"

# ------------------------------------------------------------------
# 3. Download SwissProt BLAST database
# ------------------------------------------------------------------
echo ""
echo "Downloading SwissProt BLAST database (~600 MB)..."
echo "This is a one-time download."

# Try update_blastdb.pl first (ships with BLAST+), fall back to wget
if command -v update_blastdb.pl &>/dev/null; then
    echo "Using update_blastdb.pl..."
    update_blastdb.pl --decompress swissprot
elif command -v wget &>/dev/null; then
    echo "update_blastdb.pl not found, using wget..."
    wget -q --show-progress \
        "https://ftp.ncbi.nlm.nih.gov/blast/db/swissprot.tar.gz" \
        -O swissprot.tar.gz
    tar -xzf swissprot.tar.gz
    rm swissprot.tar.gz
elif command -v curl &>/dev/null; then
    echo "update_blastdb.pl not found, using curl..."
    curl -# -L \
        "https://ftp.ncbi.nlm.nih.gov/blast/db/swissprot.tar.gz" \
        -o swissprot.tar.gz
    tar -xzf swissprot.tar.gz
    rm swissprot.tar.gz
else
    echo "ERROR: No download tool found (tried update_blastdb.pl, wget, curl)."
    exit 1
fi

# ------------------------------------------------------------------
# 4. Verify
# ------------------------------------------------------------------
echo ""
echo "Verifying database..."
if ls swissprot.p?? &>/dev/null || ls swissprot.phr &>/dev/null; then
    echo "SwissProt DB files:"
    ls -lh swissprot.* 2>/dev/null | head -10
else
    echo "WARNING: Expected swissprot.* files not found in $DB_DIR"
    echo "Contents of $DB_DIR:"
    ls -lh
    exit 1
fi

# ------------------------------------------------------------------
# 5. Quick self-test
# ------------------------------------------------------------------
echo ""
echo "Running quick BLAST self-test..."
TEST_FASTA=$(mktemp --suffix=.fasta)
echo ">test" > "$TEST_FASTA"
echo "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL" >> "$TEST_FASTA"

if blastp -query "$TEST_FASTA" -db "$DB_DIR/swissprot" -max_target_seqs 1 \
          -outfmt 6 -evalue 1e-3 -num_threads 1 &>/dev/null; then
    echo "BLAST self-test passed."
else
    echo "WARNING: BLAST self-test produced no output (DB may need indexing)."
fi
rm -f "$TEST_FASTA"

# ------------------------------------------------------------------
# 6. Done
# ------------------------------------------------------------------
echo ""
echo "=== Setup complete ==="
echo "Local BLAST+ will be used automatically by the benchmark."
echo "Run: python scripts/run_benchmark.py"
echo ""
echo "If you move the DB, set: export BLAST_LOCAL_DB=/path/to/swissprot"
