#!/bin/bash

fasta_path=$1
cpu=$2

# for relative file paths
script_path="$(readlink -f "$0")"
script_dir="$(dirname "$script_path")"

db0_path="${script_dir}/../msa_database/db/Rfam.cm"
db1_path="${script_dir}/../msa_database/db/rnacentral.fasta"
db2_path="${script_dir}/../msa_database/db/nt"
db0to1_path="${script_dir}/../msa_database/db/rfam_annotations.tsv.gz"
db0to2_path="${script_dir}/../msa_database/db/Rfam.full_region.gz"

echo "MSA search started."

./rmsa/rMSA.pl "${fasta_path}" -db0="${db0_path}" \
                               -db1="${db1_path}" \
                               -db2="${db2_path}" \
                               -db0to1="${db0to1_path}" \
                               -db0to2="${db0to2_path}" \
                               -cpu=$cpu

echo "MSA search completed."
