## Installation

Ensure you have the required Python packages installed:

```bash
pip install -r requirements.txt
```
#Learned Filter
This script adds learned filter probabilities to JSON annotation file and saves a new updated copy.

1. Run for human labels
```bash
python add_itm_probs_to_json.py
```
2. Run for Gemini labels
```bash
python add_itm_probs_to_json.py \
  --input_json /projectnb/ivc-ml/maxwh/code/labeling_effort/filter/gemini_labels.json \
  --output_json gemini_labels_with_itm_probs.json \
  --events_json gemini_labels_itm_events.jsonl \
  --snapshot_every 25
```
