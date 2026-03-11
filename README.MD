# Summary
This repository contains my exploratory work building automated data aggregation pipelines on known dataset. Eventually I'll be applying what I've learned here toward the [UNIFIED project](https://www.unified-project.eu/), but for now I'm simply exploring LLMs/agents' ability to build data aggregation pipelines on public datasets. 

Output datasets aren't included because they exceed GitHub's file storage limit.

# Covered Datasets
## UCI Opportunity Dataset
The [UCI Opportunity Dataset](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition) contains data from wearable, object, and ambient sensors that track subjects as they perform activities of daily living. 

I started on this dataset because it was well-used and we want to incorporate wearable technology. I experimented with Claude Code's (henceforth "CC") ability to identify locomotion status (one of 'sit', 'stand', 'walk', 'lie') and (to a lesser extent) arm activity with varying levels of context and with/without access to the ground truth labels. I found that CC can do fairly well without ground truth labels, but results were inherently contaminated because Claude Code could detect that this was the Opportunity dataset even without file names.

Files/folders: 
- `notes.MD` contains my notes on different iterations of the attempt to get CC to understand the data. 
- `prompts` contains my first set of attempts, where I tried various hand-crafted prompts to get CC to understand the data
- `pipelines` contains my second set of attempts, where I tried to automate the construction of the prompt to avoid hand-crafting
- `analysis.ipynb` contains some analysis of results from different attempts (not most of them)
- `secrets` contains the UCI data with Locomotion labels, which CC should not have access to
The remaining files (`convert_to_parquet.py`, `rename_acc_cols.py`, `utils.py`) are utility functions

## ScientISST MOVE
The [ScientISST MOVE dataset](https://physionet.org/content/scientisst-move-biosignals/1.0.1/) is a less well-known dataset that also contains sensor information annotated by the activities people are performing at the time. I started using this dataset in the hopes that CC would be less able to recognize it, and therefore possibly cheat on the task. 