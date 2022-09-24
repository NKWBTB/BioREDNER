# BioREDNER
NER on BioRED dataset

# Requirement
- [Stanza](https://stanfordnlp.github.io/stanza/)

# Files

- ``config.py``: dataset path configuration
- ``convert_to_bio.py``: preprocessing python script
- ``stat.py``: python script to check dataset statistics
- ``train.sh``: training bash script

# Guideline
1. Download and unzip the [BioRED](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/) dataset.

2. Config the ``config.py``, set the path of the dataset folder.

3. Follow the guides provided by ``stanza-train``,  config ``stanza-train/config.sh`` for the path of the processed dataset path and run the following to set environment variables:
```bash
source config.sh
```

4. Run the following to preprocess the dataset (tokenize and convert to BIO scheme) for stanza training ultility:
```bash
python3 convert_to_bio.py
```

5. Config ``train.sh`` for hyperparameters and run the following to train and evaluate the model:
```bash
./train.sh
```