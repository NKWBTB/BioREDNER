import json
import os
import numpy as np
import config as CFG

if __name__ == '__main__':
    data_file = os.path.join(CFG.DATASET_FOLDER, CFG.TRAIN_FILE)

    with open(data_file, "r", encoding="UTF-8") as f:
        data = json.load(f)

    entity_dict = {}
    
    text_length = []
    text_numword = []

    for doc in data["documents"]:
        for passage in doc["passages"]:
            text_length.append(len(passage["text"]))
            text_numword.append(len(passage["text"].split()))

            for annot in passage["annotations"]:
                entity_type = annot["infons"]["type"]
                entity_mention = annot["text"]
                entity_dict.setdefault(entity_type, set()).add(entity_mention)

    print(np.max(text_length), np.min(text_length), np.mean(text_length))
    print(np.max(text_numword), np.min(text_numword), np.mean(text_numword))
    print(len(entity_dict))
    
    for etype, eset in entity_dict.items():
        print(etype, len(eset))