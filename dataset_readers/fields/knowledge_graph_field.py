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
                 include_in_vocab: bool = True,
                 max_table_tokens: int = None) -> None:

        # define 8 feature extract function for extracting features when generating the linking_features.
        # The code of these 8 function is in KnowledgeGraphField, such as : "_number_token_match" and "_exact_token_match"
        # These function calculate the relationship between knowledge_graph and utterance_tokens.
        feature_extractors = feature_extractors if feature_extractors is not None else [
                'number_token_match',
                'exact_token_match',
                'contains_exact_token_match',
                'lemma_match',
                'contains_lemma_match',
                'edit_distance',
                'span_overlap_fraction',
                'span_lemma_overlap_fraction',
                ]
        # linking_features[0] is first node
        # linking_features[0][0] is first node + first utterance token
        # linking_features[0][0][0] is first feature_extractors(first node, first utterance token)
        # The dim of linking_features is: number_node_in_graph * number_utterance_token * number_feature_extractors
        # So, every method in feature_extractors return one value. Their input are the same as:(one_node, one_utterance_token)

        super().__init__(knowledge_graph, utterance_tokens, token_indexers,
                         tokenizer=tokenizer, feature_extractors=feature_extractors, entity_tokens=entity_tokens,
                         linking_features=linking_features, include_in_vocab=include_in_vocab,
                         max_table_tokens=max_table_tokens)

        # we will get a linking_features before we run this method.
        # Original self.linking_features is generated from super().__init__
        # But I think _compute_related_linking_features is useless so I add this:
        I_Think_It_Useless = self.linking_features # Check useless ???
        self.linking_features = self._compute_related_linking_features(self.linking_features)
        assert I_Think_It_Useless == self.linking_features # Check useless ???
         

        # hack needed to fix calculation of feature extractors in the inherited as_tensor method
        self._feature_extractors = feature_extractors * 2 #??????


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
