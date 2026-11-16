# NBDC_tools

Small utilities for working with NBDC datasets (HBCD, ABCD, ...).

## Scripts

- `summarize_bids_downloads.py` — summarize subjects/sessions downloaded from BIDS-style
  datasets across `rawdata` and each derivative pipeline. Writes long/summary/coverage CSVs and plots.

```
python summarize_bids_downloads.py /nbdc/hbcd_2.1/hbcd -o hbcd_summary
```
