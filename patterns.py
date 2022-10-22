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
        norm_id = {}
        mention_dict = {}
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
                    if not id in norm_id:
                        norm_id[id] = str(len(norm_id))
                    new_id.append(norm_id[id])
                    if not id in mention_dict:
                        mention_dict[id] = typedict[entity_type]+ "_" + norm_id[id]
                
                # Memorize all the substitution in a dictionary
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
        
        for relation in doc["relations"]:
            head_entity = mention_dict[relation["infons"]["entity1"]]
            tail_entity = mention_dict[relation["infons"]["entity2"]]
            
            def substitude2nodes(token):
                # Case 1: The word is not a substitution
                if not token.text in mention_set:
                    return [token.i] 
                # Case 2: The word is a substitution
                entity_info = token.text.split('_')
                entity_type = entity_info[0]
                entity_ids = entity_info[1:]
                possible_entities = [entity_type+'_'+id for id in entity_ids]
                # Case 2-1: This word appeared as head/tail entity
                if head_entity in possible_entities and tail_entity in possible_entities:
                    return [head_entity, tail_entity]
                if head_entity in possible_entities: return [head_entity]
                if tail_entity in possible_entities: return [tail_entity]
                # Case 2-2: This word is not a part of head/tail entity
                return [token.i]

            edges = []
            def addedge(sources, targets):
                for u in sources:
                    for v in targets:
                        edges.append((u, v))
            
            last_root = -1
            for token in spacy_doc:
                source_node = substitude2nodes(token)
                # For each pair of adjacent sentences in one document, add an edge between their roots 
                if str(token.dep_) == "ROOT":
                    if last_root != -1:
                        addedge(last_root, source_node)
                    last_root = source_node

                for child in token.children:
                    target_node = substitude2nodes(child)  
                    addedge(source_node, target_node)
            graph = nx.Graph(edges)
            
            # 2. Find the shortest path from the head entity to the tail entity
            try:
                paths = nx.all_shortest_paths(graph, source=head_entity, target=tail_entity)
                for path in paths:
                    pattern = []
                    for node in path:
                        if type(node) == int:
                            if not spacy_doc[node].text in mention_set:
                                pattern.append(spacy_doc[node].text.lower())
                            else:
                                pattern.append(spacy_doc[node].text.split("_")[0])
                        else:
                            pattern.append(node.split("_")[0])
                    
                    #3. Count the frequency of the paths. 
                    pattern_text = " ".join(pattern)
                    if not pattern_text in pattern_count:
                        pattern_count[pattern_text] = 1
                    else:
                        pattern_count[pattern_text] += 1
            except:
                print(head_entity, tail_entity)
                assert False
                

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