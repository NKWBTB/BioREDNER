import json
import os
import config as CFG
import spacy
import networkx as nx
import csv
from tqdm import tqdm

typedict = {
    "GeneOrGeneProduct": "GENE",
    "DiseaseOrPhenotypicFeature": "DISEASE",
    "ChemicalEntity": "CHEMICAL",
    "SequenceVariant": "VARIANT"
}

nlp = spacy.load("en_core_web_sm")

def mine_pattern(data, pattern_count):
    for doc in tqdm(data["documents"]):
        # 1. replace the entity mentions with their entity types (GENE, DISEASE, CHEMICAL, VARIANT) and IDs
        id2norm = {}
        norm2id = {}
        mention_set = set()
        sents = []
        for passage in doc["passages"]:
            text = passage["text"]
            edits = []
            for annot in passage["annotations"]:
                entity_id = annot["infons"]["identifier"]
                entity_type = annot["infons"]["type"]
                if not entity_type in typedict: continue

                # Normalize ids (prevent tokenizer split)
                ids = entity_id.split(",")
                new_id = []
                for id in ids:
                    if not id in id2norm:
                        idx = str(len(id2norm))
                        id2norm[id] = idx
                        norm2id[idx] = id
                    new_id.append(id2norm[id])
                
                # Memorize all the substitution in a set
                substitude = typedict[entity_type]+ "_" + "_".join(new_id)
                mention_set.add(substitude)
                
                for loc in annot["locations"]:
                    start = loc["offset"] - passage["offset"]
                    end = start + loc["length"]
                    edits.append((start, end, substitude))
            
            edits = sorted(edits, reverse=True)
            for start, end, substitude in edits:
                text = text[:start] + ' ' + substitude + ' ' + text[end:]

            sents.append(' '.join(text.split()))
        
        # Apply spacy dependency parser for each sentence. 
        doc_text = " ".join(sents)
        spacy_doc = nlp(doc_text)
        
        id2node = {}
        # Map identifiers to the node index in graph
        def token2node(token):
            if not token.text in mention_set:
                return
            entity_info = token.text.split('_')
            entity_normids = entity_info[1:]
            for norm_id in entity_normids:
                id = norm2id[norm_id]
                id2node.setdefault(id, set()).add(token.i)
        
        # 2. Form document graphs for each document
        edges = []
        last_root = -1
        for token in spacy_doc:
            token2node(token)
            # For each pair of adjacent sentences in one document, add an edge between their roots 
            if str(token.dep_) == "ROOT":
                if last_root != -1: edges.append((last_root, token.i))
                last_root = token.i

            for child in token.children:
                edges.append((token.i, child.i))
        graph = nx.Graph(edges)

        # 3. Find the shortest path from the head entity to the tail entity
        for relation in doc["relations"]:
            head_entity = id2node[relation["infons"]["entity1"]]
            tail_entity = id2node[relation["infons"]["entity2"]]

            shortest_paths = {}
            for u in head_entity:
                for v in tail_entity:
                    paths = nx.all_shortest_paths(graph, source=u, target=v)
                    for path in paths:
                        shortest_paths.setdefault(len(path), []).append(path.copy())
            
            min_len = min(shortest_paths.keys())
            for path in shortest_paths[min_len]:
                pattern = []
                # Truncate head and tail
                path = path[1:-1]
                for node in path:
                    if spacy_doc[node].text in mention_set:
                        pattern.append(spacy_doc[node].text.split("_")[0])
                    else:
                        pattern.append(spacy_doc[node].text.lower())
                
                # 4. Count the frequency of the paths. 
                pattern_text = " ".join(pattern)
                if len(pattern_text) == 0: pattern_text = " "                    
                if not pattern_text in pattern_count:
                    pattern_count[pattern_text] = 1
                else:
                    pattern_count[pattern_text] += 1
                

if __name__ == '__main__':
    pattern_count = {}
    dataset = {"train": CFG.TRAIN_FILE, "dev": CFG.DEV_FILE, "test": CFG.TEST_FILE}

    for data_split, data_file in dataset.items():
        print(f'Processing split:{data_split}')
        data_file = os.path.join(CFG.DATASET_FOLDER, data_file)

        with open(data_file, "r", encoding="UTF-8") as f:
            data = json.load(f)

        mine_pattern(data, pattern_count)

    # Sort the paths by their frequencies (high to low)
    freq_pattern = [(freq, pattern) for pattern, freq in pattern_count.items()]
    freq_pattern = sorted(freq_pattern, reverse=True)

    with open("output.csv", "w", encoding="UTF-8") as f:
        writer = csv.writer(f, delimiter='\t')
        for freq, pattern in freq_pattern:
            writer.writerow([pattern, freq])