# """
# ``KnowledgeGraphField`` is a ``Field`` which stores a knowledge graph representation.
# """
# from typing import List, Dict

# from allennlp.data import TokenIndexer, Tokenizer
# from allennlp.data.fields.knowledge_graph_field import KnowledgeGraphField
# from allennlp.data.tokenizers.token import Token
# from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph


# class SpiderKnowledgeGraphField(KnowledgeGraphField):
#     """
#     This implementation calculates all non-graph-related features (i.e. no related_column),
#     then takes each one of the features to calculate related column features, by taking the max score of all neighbours
#     """
#     def __init__(self,
#                  knowledge_graph: KnowledgeGraph,
#                  utterance_tokens: List[Token],
#                  token_indexers: Dict[str, TokenIndexer],
#                  tokenizer: Tokenizer = None,
#                  feature_extractors: List[str] = None,
#                  entity_tokens: List[List[Token]] = None,
#                  linking_features: List[List[List[float]]] = None,
#                  include_in_vocab: bool = True,     # Although default is True, it will send a False to here from calling
#                  max_table_tokens: int = None,
#                  conceptnet = None,
#                  concept_similarity = None) -> None:

#         # The shape of linking_features will be:
#         # [ <len of KnowledgeGraph.entities (graph node number)> * <len of utterance_tokens> * <len of feature_extractors> ]
#         # define 8 feature extract function for extracting features when generating the linking_features.
#         # The code of these 8 function is in KnowledgeGraphField, such as : "_number_token_match" and "_exact_token_match"
#         # These function calculate the relationship between knowledge_graph and utterance_tokens.
#         # ------
#         # linking_features is created by feature_extractors. feature_extractors are a function list.
#         # linking_features[0] is first node
#         # linking_features[0][0] is first node + first utterance token
#         # linking_features[0][0][0] is first feature_extractors(first node, first utterance token)
#         # The dim of linking_features is: number_node_in_graph * number_utterance_token * number_feature_extractors
#         # So, every method in feature_extractors return one value. Their input are the same as:(one_node, one_utterance_token)
#         # It will call: feature_extractor(entity, entity_text, token, token_index, self.utterance_tokens)
#         # entity is name of node. entity_text is value of node(or text of node). token is token of utterance. token_index is the token index from utterance. self.utterance_tokens is all token of utterance. 
#         feature_extractors = feature_extractors if feature_extractors is not None else [
#                 'number_token_match',           # If input node text is a number and match to the input token of question, then return 1.
#                 'exact_token_match',            # If input node text match to the input token of question, then return 1.
#                 'contains_exact_token_match',   # If input node text contain the input token of question, then return 1.
#                 'lemma_match',                  # If lemma of input node text match to the lemma of input token of question, then return 1.
#                 'contains_lemma_match',         # If lemma of input node text contain the lemma of input token of question, then return 1.
#                 'edit_distance',
#                 'span_overlap_fraction',        # I give you some example: span_overlap_fraction(?, 'born state', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 1.
#                                                 # span_overlap_fraction(?, 'born', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 1.
#                                                 # span_overlap_fraction(?, 'state', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 0.
#                                                 # span_overlap_fraction(?, 'state born', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 1.
#                                                 # span_overlap_fraction(?, 'born states', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 1/2=0.5.
#                                                 # span_overlap_fraction(?, 'born state aaa', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 2/3=0.66.
#                                                 # span_overlap_fraction(?, 'born state', ?, 5, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 1.
#                                                 # If the self.utterance_tokens[token_index] do not appear in entity_text, return 0.
#                                                 # Else: It will check how many unique words appearing before, after and in token_index of self.utterance_tokens. And then return: this_number / len(entity_text)
#                 'span_lemma_overlap_fraction',  # span_overlap_fraction(?, 'born states', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']) return 1/2=0.5. But span_lemma_overlap_fraction return 1.
#                 ]

#         super().__init__(knowledge_graph, utterance_tokens, token_indexers,
#                          tokenizer=tokenizer, feature_extractors=feature_extractors, entity_tokens=entity_tokens,
#                          linking_features=linking_features, include_in_vocab=include_in_vocab,
#                          max_table_tokens=max_table_tokens)

#         # we will get a linking_features before we run this method.
#         # Original self.linking_features is generated from super().__init__
#         # But I think _compute_related_linking_features is useless so I add this:
#         self.linking_features = self._compute_related_linking_features(self.linking_features)
         

#         # hack needed to fix calculation of feature extractors in the inherited as_tensor method
#         self._feature_extractors = feature_extractors * 2 # I think here is also useless ???


#     # Here is not the overrides of "_compute_linking_features" that is the original function for calculating the linking_features. 
#     def _compute_related_linking_features(self,
#                                           non_related_features: List[List[List[float]]]) -> List[List[List[float]]]:
#         linking_features = non_related_features
#         entity_to_index_map = {}
#         for entity_id, entity in enumerate(self.knowledge_graph.entities):
#             entity_to_index_map[entity] = entity_id

#         for entity_id, (entity, entity_text) in enumerate(zip(self.knowledge_graph.entities, self.entity_texts)):
#             for token_index, token in enumerate(self.utterance_tokens):
#                 entity_token_features = linking_features[entity_id][token_index]
#                 for feature_index, feature_extractor in enumerate(self._feature_extractors):
#                     neighbour_features = []
#                     for neighbor in self.knowledge_graph.neighbors[entity]:
#                         # we only care about table/columns relations here, not foreign-primary
#                         if entity.startswith('column') and neighbor.startswith('column'):
#                             continue
#                         neighbor_index = entity_to_index_map[neighbor]
#                         neighbour_features.append(non_related_features[neighbor_index][token_index][feature_index])

#                     entity_token_features.append(max(neighbour_features))

#         return linking_features


"""
``KnowledgeGraphField`` is a ``Field`` which stores a knowledge graph representation.
"""
from typing import List, Dict

from allennlp.data import TokenIndexer, Tokenizer
from allennlp.data.fields.knowledge_graph_field import KnowledgeGraphField
from allennlp.data.tokenizers.token import Token
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph


import numpy as np
import copy


STOP_WORDS = {"", "", "all", "being", "-", "over", "through", "yourselves", "its", "before",
              "hadn", "with", "had", ",", "should", "to", "only", "under", "ours", "has", "ought", "do",
              "them", "his", "than", "very", "cannot", "they", "not", "during", "yourself", "him",
              "nor", "did", "didn", "'ve", "this", "she", "each", "where", "because", "doing", "some", "we", "are",
              "further", "ourselves", "out", "what", "for", "weren", "does", "above", "between", "mustn", "?",
              "be", "hasn", "who", "were", "here", "shouldn", "let", "hers", "by", "both", "about", "couldn",
              "of", "could", "against", "isn", "or", "own", "into", "while", "whom", "down", "wasn", "your",
              "from", "her", "their", "aren", "there", "been", ".", "few", "too", "wouldn", "themselves",
              ":", "was", "until", "more", "himself", "on", "but", "don", "herself", "haven", "those", "he",
              "me", "myself", "these", "up", ";", "below", "'re", "can", "theirs", "my", "and", "would", "then",
              "is", "am", "it", "doesn", "an", "as", "itself", "at", "have", "in", "any", "if", "!",
              "again", "'ll", "no", "that", "when", "same", "how", "other", "which", "you", "many", "shan",
              "'t", "'s", "our", "after", "most", "'d", "such", "'m", "why", "a", "off", "i", "yours", "so",
              "the", "having", "once"}

class TokenList():
    def __init__(self, column_token):
        self.column_token = column_token

    def __next__(self):
        return self.column_token.__next__()

    def __iter__(self):
        return self.column_token.__iter__()

    def __len__(self):
        # tok_set = set()
        # for tok in self.column_token:
        #     if tok.text.lower() not in ['id', "*"]:
        #         tok_set.add(tok.lemma_)
        # len_v = len(tok_set)
        # if len_v > 0:
        #     return len_v
        # elif len(self.column_token) > 0:
        #     return len(self.column_token)
        # return 1
        return len(self.column_token)

class SpiderKnowledgeGraphField(KnowledgeGraphField):
    """
    This implementation calculates all non-graph-related features (i.e. no related_column),
    then takes each one of the features to calculate related column features, by taking the max score of all neighbours
    """
    def __init__(self,
                 knowledge_graph: KnowledgeGraph,
                 utterance_tokens: List[Token],
                 token_indexers: Dict[str, TokenIndexer],
                 tokenizer: Tokenizer = None,
                 feature_extractors: List[str] = None,
                 entity_tokens: List[List[Token]] = None,
                 linking_features: List[List[List[float]]] = None,
                 include_in_vocab: bool = True,     # Although default is True, it will send a False to here from calling
                 max_table_tokens: int = None,
                 conceptnet = None,
                 concept_similarity = None) -> None:

        column_tokens = []
        for i in entity_tokens:
            column_tokens.append(TokenList(i))
        entity_tokens = column_tokens

        def add_special_token_to_utterance(column_tokens):
            # new_token = []
            specail_list = ["year","name"]
            specail_already = [False,False]
            for col in column_tokens:
                for i, tok in enumerate(specail_list) :
                    if tok == col.lemma_:
                        specail_already[i] = True
            
            name_start_token = ["what","give","tell","show","which","find"]

            for i, col in enumerate(column_tokens):
                if col.text.isdigit():
                    digit = int(col.text)
                    if digit > 1700 and digit < 2100 and not specail_already[0]:
                        column_tokens[i] = Token(text="year",lemma="year",tag = "NN")
                        # new_token.append(Token(text="year",lemma="year",tag = "NN"))
                        specail_already[0] = True
                elif col.text == "each" or (i==0 and col.lemma_ in name_start_token) and not specail_already[1]:
                    column_tokens[i] = Token(text="name",lemma="name",tag = "NN")
                    # new_token.append(Token(text="name",lemma="name",tag = "NN"))
                    specail_already[1] = True
            # column_tokens.extend(new_token)
            return column_tokens
        utterance_tokens = add_special_token_to_utterance(utterance_tokens)

        # The shape of linking_features will be:
        # [ <len of KnowledgeGraph.entities (graph node number)> * <len of utterance_tokens> * <len of feature_extractors> ]
        # define 8 feature extract function for extracting features when generating the linking_features.
        # The code of these 8 function is in KnowledgeGraphField, such as : "_number_token_match" and "_exact_token_match"
        # These function calculate the relationship between knowledge_graph and utterance_tokens.
        # ------
        # linking_features is created by feature_extractors. feature_extractors are a function list.
        # linking_features[0] is first node
        # linking_features[0][0] is first node + first utterance token
        # linking_features[0][0][0] is first feature_extractors(first node, first utterance token)
        # The dim of linking_features is: number_node_in_graph * number_utterance_token * number_feature_extractors
        # So, every method in feature_extractors return one value. Their input are the same as:(one_node, one_utterance_token)
        # It will call: feature_extractor(entity, entity_text, token, token_index, self.utterance_tokens)
        # entity is name of node. entity_text is value of node(or text of node). token is token of utterance. token_index is the token index from utterance. self.utterance_tokens is all token of utterance. 
        feature_extractors = feature_extractors if feature_extractors is not None else [
            'number_token_match',#'number_token_match',           # If input node text is a number and match to the input token of question, then return 1.
            'span_overlap_fraction',#'exact_token_match',            # If input node text match to the input token of question, then return 1.
                'contains_exact_token_match',#'contains_exact_token_match',   # If input node text contain the input token of question, then return 1.
                'span_lemma_overlap_fraction',#'lemma_match',                  # If lemma of input node text match to the lemma of input token of question, then return 1.
                'contains_lemma_match',#'contains_lemma_match',         # If lemma of input node text contain the lemma of input token of question, then return 1.
                'edit_distance',#'edit_distance',
                'span_overlap_fraction',        # I give you some example: span_overlap_fraction(?, 'born state', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 1.
                                                # span_overlap_fraction(?, 'born', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 1.
                                                # span_overlap_fraction(?, 'state', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 0.
                                                # span_overlap_fraction(?, 'state born', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 1.
                                                # span_overlap_fraction(?, 'born states', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 1/2=0.5.
                                                # span_overlap_fraction(?, 'born state aaa', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 2/3=0.66.
                                                # span_overlap_fraction(?, 'born state', ?, 5, ['list', 'the', 'name', ',', 'born', 'state', 'and']); It will return 1.
                                                # If the self.utterance_tokens[token_index] do not appear in entity_text, return 0.
                                                # Else: It will check how many unique words appearing before, after and in token_index of self.utterance_tokens. And then return: this_number / len(entity_text)
                'span_lemma_overlap_fraction',  # span_overlap_fraction(?, 'born states', ?, 4, ['list', 'the', 'name', ',', 'born', 'state', 'and']) return 1/2=0.5. But span_lemma_overlap_fraction return 1.
                ]

        super().__init__(knowledge_graph, utterance_tokens, token_indexers,
                         tokenizer=tokenizer, feature_extractors=feature_extractors, entity_tokens=entity_tokens,
                         linking_features=linking_features, include_in_vocab=include_in_vocab,
                         max_table_tokens=max_table_tokens)

        if conceptnet:
            self._concept_word = conceptnet
            scores = np.array(self.linking_features)
            scores2 = self.conceptnet_new_match(scores[:,:,1], utterance_tokens, entity_tokens)
            for i in range(len(entity_tokens)):
                for j in range(len(utterance_tokens)):
                    self.linking_features[i][j][0] = scores2[i,j]
                    self.linking_features[i][j][5] = scores2[i,j]
            self._concept_word = None
        
        if concept_similarity:
            pass

        # we will get a linking_features before we run this method.
        # Original self.linking_features is generated from super().__init__
        # But I think _compute_related_linking_features is useless so I add this:
        # Original linking_features shape is [? * ? * len(feature_extractors)]
        # However, _compute_related_linking_features make the it become [? * ? * len(feature_extractors)*2]
        self.linking_features = self._compute_related_linking_features(self.linking_features)         

        # hack needed to fix calculation of feature extractors in the inherited as_tensor method
        # I think here is useless, but it can tell us that the shape of self.linking_features is [? * ? * len(feature_extractors)*2]
        self._feature_extractors = feature_extractors * 2 


    # Here is not the overrides of "_compute_linking_features" that is the original function for calculating the linking_features. 
    def _compute_related_linking_features(self,
                                          non_related_features: List[List[List[float]]]) -> List[List[List[float]]]:
        linking_features = non_related_features
        entity_to_index_map = {}
        for entity_id, entity in enumerate(self.knowledge_graph.entities):
            entity_to_index_map[entity] = entity_id

        for entity_id, (entity, entity_text) in enumerate(zip(self.knowledge_graph.entities, self.entity_texts)):
            for token_index, token in enumerate(self.utterance_tokens):
                entity_token_features = linking_features[entity_id][token_index]
                for feature_index, feature_extractor in enumerate(self._feature_extractors):
                    neighbour_features = []
                    for neighbor in self.knowledge_graph.neighbors[entity]:
                        # we only care about table/columns relations here, not foreign-primary
                        if entity.startswith('column') and neighbor.startswith('column'):
                            continue
                        neighbor_index = entity_to_index_map[neighbor]
                        neighbour_features.append(non_related_features[neighbor_index][token_index][feature_index])

                    entity_token_features.append(max(neighbour_features))

        return linking_features




    def get_related_word(self,words,mini_weight=0):
        result = set()
        for word in words:
            word = word.lower()
            if word in self._concept_word.keys():
                re_words = self._concept_word[word]
                for w in re_words:
                    if w[0] not in words and w[1]>mini_weight:
                        result.add(w[0])
        return list(result)

    def update_scores_using_conceptnet(self, idx_utter, scores, column_tokens, conceptnets):
        success = False
        for wi, related_w in enumerate(conceptnets):
            for ci, cols in enumerate(column_tokens):
                for col in cols:
                    if col.text != "*" and col.lemma_ == related_w or col.text == related_w:
                        scores[ci][idx_utter] = 1
                        success = True
        return scores, success

    def new_update_scores_using_conceptnet(self, column_tokens, conceptnets):
        success = False
        re_score = 0
        for wi, related_w in enumerate(conceptnets):
            if column_tokens.lemma_ == related_w or column_tokens.text == related_w:
                re_score = 1
                success = True
                break
        return re_score, success
    
    def conceptnet_match(self, old_scores,  utter_tokens, column_tokens):
        # calc the match scores:
        # scores = np.array(scores)
        scores = np.zeros([len(column_tokens),len(utter_tokens)])
        sum_s = old_scores.sum(0)
        for i, word in enumerate(utter_tokens):
            if sum_s[i] > 0 or word.lemma_ in STOP_WORDS or word.lemma_ == "number":
                continue
            conceptnet_text = self.get_related_word([word.text])
            scores, match = self.update_scores_using_conceptnet(i, scores, column_tokens, conceptnet_text)
            if not match and word.text != word.lemma_:
                conceptnet_lemma = self.get_related_word([word.lemma_])
                scores, match = self.update_scores_using_conceptnet(i, scores, column_tokens, conceptnet_lemma)
            if not match:
                conceptnet_text = self.get_related_word(conceptnet_text,0.5)
                scores, match = self.update_scores_using_conceptnet(i, scores, column_tokens, conceptnet_text)
            if not match and word.text != word.lemma_:
                conceptnet_lemma = self.get_related_word(conceptnet_lemma,0.5)
                scores, match = self.update_scores_using_conceptnet(i, scores, column_tokens, conceptnet_lemma)
        return scores

    def length_of_col(self, cols):
        len_ = len(cols)
        for col in cols:
            if col.lemma_ in STOP_WORDS or col.text == "*":
                len_ -= 1
        if len_ <= 0:
            return 1
        return len_

    def conceptnet_new_match(self, old_scores,  utter_tokens, column_tokens):
        # calc the match scores:
        # scores = copy.deepcopy(old_scores)
        scores = np.zeros([len(column_tokens),len(utter_tokens)])
        # sum_every_u = old_scores.sum(0)
        max_every_col = old_scores.max(1)
        for i, col in enumerate(column_tokens):
            if max_every_col[i] >= 1:
                continue
            ss = np.zeros([len(col.column_token),len(utter_tokens)])
            for ii, col_w in enumerate(col.column_token):
                if col_w.lemma_ in STOP_WORDS or col_w.text == "*":
                    continue
                for j, word in enumerate(utter_tokens):
                    if col_w.lemma_ == word.lemma_:
                        ss[ii,j] = 1
                        break
                    elif word.lemma_ in STOP_WORDS or word.lemma_ == "number":
                        continue
                    conceptnet_text = self.get_related_word([word.text])
                    ss[ii,j], _ = self.new_update_scores_using_conceptnet( col_w, conceptnet_text)
            final_s = ss.max(1)
            final_s = final_s.sum()/self.length_of_col(col.column_token)
            for_idx = ss.max(0)
            scores[i,for_idx==1]=final_s
        return scores
    
    def conceptnet_similarity(self, old_scores,  utter_tokens, column_tokens):
        scores = np.zeros([len(column_tokens),len(utter_tokens)])
        scores = copy.deepcopy(old_scores)