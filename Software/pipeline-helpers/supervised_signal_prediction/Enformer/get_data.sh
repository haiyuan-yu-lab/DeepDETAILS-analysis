#/usr/bin/env bash

gsutil -u $GCP_PROJECT -m cp \
  "gs://basenji_barnyard/data/human/tfrecords/test-0-0.tfr" \
  "gs://basenji_barnyard/data/human/tfrecords/test-0-1.tfr" \
  "gs://basenji_barnyard/data/human/tfrecords/test-0-2.tfr" \
  "gs://basenji_barnyard/data/human/tfrecords/test-0-3.tfr" \
  "gs://basenji_barnyard/data/human/tfrecords/test-0-4.tfr" \
  "gs://basenji_barnyard/data/human/tfrecords/test-0-5.tfr" \
  "gs://basenji_barnyard/data/human/tfrecords/test-0-6.tfr" \
  "gs://basenji_barnyard/data/human/tfrecords/test-0-7.tfr" \
  .
gsutil -u $GCP_PROJECT -m cp \
  "gs://basenji_barnyard/data/human/sequences.bed" \
  "gs://basenji_barnyard/data/human/statistics.json" \
  "gs://basenji_barnyard/data/human/targets.txt" \
  .
gsutil -u $GCP_PROJECT -m cp gs://basenji_barnyard/hg38.ml.fa.gz .
gunzip hg38.ml.fa.gz
