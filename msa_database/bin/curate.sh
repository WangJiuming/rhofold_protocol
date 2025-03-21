#!/bin/bash
FILE=`readlink -e $0`
bindir=`dirname $FILE`
rootdir=`dirname $bindir`
cd "$rootdir/db"

echo "extracting fasta"
if [ -s "rnacentral_species_specific_ids.fasta.gz" ];then
    zcat rnacentral_species_specific_ids.fasta.gz|grep -ohP "^\S+"| $bindir/fastaNA -| $bindir/catRNAcentral - rnacentral.fasta rnacentral.tsv
    rm rnacentral_species_specific_ids.fasta.gz
fi

echo "running makeblastdb"
$bindir/makeblastdb -in rnacentral.fasta -parse_seqids -hash_index -dbtype nucl

echo "unzipping Rfam"
if [ -s "Rfam.cm.gz" ];then
    gzip -d Rfam.cm.gz
fi
rm Rfam.cm.*
$bindir/cmpress Rfam.cm

echo "extracting nt"
for filename in `ls nt*tar.gz`;do
    tar -xvf $filename
    rm $filename
done

if [ -s "nt.gz" ];then
    zcat nt.gz | grep -ohP "^\S+" | $bindir/fastaNA - > nt
    rm nt.gz
fi

echo "curation complete"

