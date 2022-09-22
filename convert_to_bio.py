from heapq import merge
import stanza
import stanza.utils.default_paths as default_paths
from stanza.utils import datasets
from stanza.utils.datasets import ner
from stanza.utils.datasets.ner.prepare_ner_dataset import convert_bio_to_json
from tqdm import tqdm
import json
import os
import config as CFG
import logging
import errno

logger = logging.getLogger("convert_to_bio")

nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)

def convert(data, dump_to=None):
    converted_data = []
    merge_count = 0
    for doc in tqdm(data["documents"]):
        for passage in doc["passages"]:
            text = passage["text"]
            tokenized_doc = nlp(text)
            tokens = tokenized_doc.sentences[0].tokens
            
            sample = []
            for token in tokens:
                token_info = {"text": token.text, "label": "O"}
                extra_token = None
                for annot in passage["annotations"]:
                    # Match by brute force
                    for loc in annot["locations"]:
                        start = loc["offset"]-passage["offset"]
                        end = start + loc["length"]
                        if token.start_char >= start and token.start_char < end:
                            if token.start_char == start:
                                token_info["label"] = "B-" + annot["infons"]["type"]
                            else:
                                token_info["label"] = "I-" + annot["infons"]["type"]
                            
                            # Test if a token is split by annotation
                            if token.end_char > end:
                                logger.warning("Token split by an annotation")
                                logger.info(doc["id"])
                                logger.info(annot)
                                logger.info(token)
                                split_point = end-token.start_char
                                token_info["text"] = token.text[:split_point]
                                extra_token = {"text": token.text[split_point:], "label": "O"}

                sample.append(token_info)
                if extra_token:
                    sample.append(extra_token)
            
            # Exceed length limit
            limit = 256
            if len(sample) > limit:
                merge_count += 1
                logger.warning("Token number exceeds limit")
                # Segment by period
                split_samples = []
                new_sample = []
                for token_info in sample:
                    new_sample.append(token_info)
                    if token_info["text"] == '.' and token_info['label'] == 'O':
                        split_samples.append(new_sample)
                        new_sample = []
                if len(new_sample) > 0: split_samples.append(new_sample)
                
                print("Sent lengths:", [len(sample) for sample in split_samples])
                # Merge by maximum space allow
                merge_samples = []
                new_sample = []
                for sample in split_samples:
                    if len(new_sample) + len(sample) <= limit:
                        new_sample.extend(sample)
                    else:
                        merge_samples.append(new_sample)
                        new_sample = []
                        new_sample.extend(sample)
                if len(new_sample) > 0: merge_samples.append(new_sample)
                converted_data.extend(merge_samples)
                print("New lengths: ", [len(sample) for sample in merge_samples])
            else:
                converted_data.append(sample)
    
    print(merge_count)

    with open(dump_to, "w", encoding="UTF-8") as f:
        for sample in converted_data:
            for token_info in sample:
                f.write(f'{token_info["text"]}\t{token_info["label"]}\n')
            f.write('\n')
       

if __name__ == '__main__':
    paths = default_paths.get_default_paths()
    output_path = os.path.join(paths["NERBASE"], CFG.DATASET)
    print(output_path)
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
        except OSError as exc: # Guard against rare conditions
            if exc.errno != errno.EEXIST:
                raise
    shortname = (CFG.LANG + "_" + CFG.DATASET).lower()

    dataset = {"train": CFG.TRAIN_FILE, "dev": CFG.DEV_FILE, "test": CFG.TEST_FILE}

    for data_split, data_file in dataset.items():
        print(f'Processing split:{data_split}')
        data_file = os.path.join(CFG.DATASET_FOLDER, data_file)

        with open(data_file, "r", encoding="UTF-8") as f:
            data = json.load(f)

        convert(data, os.path.join(output_path, '%s.%s.bio' % (shortname, data_split)))
    
    convert_bio_to_json(output_path, paths["NER_DATA_DIR"], shortname)