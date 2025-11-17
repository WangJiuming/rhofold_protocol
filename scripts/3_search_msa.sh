#!/bin/bash

fasta_path=$1

db0_path="./msa_database/db/Rfam.cm"
db1_path="./msa_database/db/rnacentral.fasta"
db2_path="./msa_database/db/nt"
db0to1_path="./msa_database/db/rfam_annotations.tsv.gz"
db0to2_path="./msa_database/db/Rfam.full_region.gz"
cpu=32

echo "MSA search started."

./rmsa/rMSA.pl "${fasta_path}" -db0="${db0_path}" \
                               -db1="${db1_path}" \
                               -db2="${db2_path}" \
                               -db0to1="${db0to1_path}" \
                               -db0to2="${db0to2_path}" \
                               -cpu=$cpu

echo "MSA search completed."
