# Preserved audit scripts
These two scripts are byte-identical copies of the source scripts run in CP05. They require the original private evidence packets and are not a fresh-data campaign.

After extracting the source final.zip and NULL_REPLAY_PACKET to directories of your choice, run:
```sh
python audits/final_independent_audit.py /path/to/extracted/final/huggingface
python audits/audit_v7_null_replay.py /path/to/extracted/NULL_REPLAY_PACKET replay_result.json
```
Source identifiers and hashes are in the manuscript supplement and verification/source_archive_receipts.json at release root. Old absolute paths in historical receipts describe this actual run, not portable installation instructions. Python 3.13.5 and NumPy 2.3.5 were used for the certified wrapper tests. The original source scripts also require their declared NumPy/SciPy dependencies.
