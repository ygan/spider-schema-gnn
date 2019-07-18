"""
``KnowledgeGraphField`` is a ``Field`` which stores a knowledge graph representation.
"""
from typing import List, Dict

from allennlp.data import TokenIndexer, Tokenizer
from allennlp.data.fields.knowledge_graph_field import KnowledgeGraphField
from allennlp.data.tokenizers.token import Token
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph


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
                 max_table_tokens: int = None) -> None:

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
                'number_token_match',           # If input node text is a number and match to the input token of question, then return 1.
                'exact_token_match',            # If input node text match to the input token of question, then return 1.
                'contains_exact_token_match',   # If input node text contain the input token of question, then return 1.
                'lemma_match',                  # If lemma of input node text match to the lemma of input token of question, then return 1.
                'contains_lemma_match',         # If lemma of input node text contain the lemma of input token of question, then return 1.
                'edit_distance',
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

        # we will get a linking_features before we run this method.
        # Original self.linking_features is generated from super().__init__
        # But I think _compute_related_linking_features is useless so I add this:
        Gan_Think_It_Useless = self.linking_features # Check useless ???
        self.linking_features = self._compute_related_linking_features(self.linking_features)
        assert Gan_Think_It_Useless == self.linking_features # Check useless ???
         

        # hack needed to fix calculation of feature extractors in the inherited as_tensor method
        self._feature_extractors = feature_extractors * 2 # I think here is also useless ???


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
