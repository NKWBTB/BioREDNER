import stanza
from tqdm import tqdm
import json
import os
import config as CFG
import logging

nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True)

def convert(data, dump_to=None):
    converted_data = []
    for doc in tqdm(data["documents"]):
        for passage in doc["passages"]:
            text = passage["text"]
            tokenized_doc = nlp(text)
            tokens = tokenized_doc.sentences[0].tokens
            
            sample = []
            for token in tokens:
                token_info = {"text": token.text, "label": "O"}
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
                                logging.warning("Token split by an annotation")
                                logging.info(doc["id"])
                                logging.info(annot)
                                logging.info(token)

                sample.append(token_info)
            converted_data.append(sample)

    with open(dump_to, "w", encoding="UTF-8") as f:
        for sample in converted_data:
            for token_info in sample:
                f.write(f'{token_info["text"]}\t{token_info["label"]}\n')
            f.write('\n')
       

if __name__ == '__main__':
    data_file = os.path.join(CFG.DATASET_FOLDER, CFG.TRAIN_FILE)

    with open(data_file, "r", encoding="UTF-8") as f:
        data = json.load(f)

    convert(data, "test.bio")