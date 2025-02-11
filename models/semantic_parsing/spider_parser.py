import difflib
import os
from functools import partial
from typing import Dict, List, Tuple, Any, Mapping, Sequence

import sqlparse
import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule, ProductionRuleArray
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, Embedding, Attention, FeedForward, \
    TimeDistributed
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util, Activation
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.states import GrammarStatelet
from torch_geometric.data import Data, Batch

from modules.gated_graph_conv import GatedGraphConv
from semparse.worlds.evaluate_spider import evaluate
from state_machines.states.rnn_statelet import RnnStatelet
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.training.metrics import Average
from overrides import overrides

from semparse.contexts.spider_context_utils import action_sequence_to_sql
from semparse.worlds.spider_world import SpiderWorld
from state_machines.states.grammar_based_state import GrammarBasedState
from state_machines.states.sql_state import SqlState
from state_machines.transition_functions.attend_past_schema_items_transition import \
    AttendPastSchemaItemsTransitionFunction
from state_machines.transition_functions.linking_transition_function import LinkingTransitionFunction


@Model.register("spider")
class SpiderParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,                   # one layer LSTM. 400 -> 200*2. 2 Because bi-direction.
                 entity_encoder: Seq2VecEncoder,            # boe is BagOfEmbeddingsEncoder. "embedding_dim": 200, "averaged": true. So here calc the average.
                 decoder_beam_search: BeamSearch,           # 10
                 question_embedder: TextFieldEmbedder,      # None pretrain embedder but trainable here
                 input_attention: Attention,                # {"type": "dot_product"},
                 past_attention: Attention,                 # {"type": "dot_product"},
                 max_decoding_steps: int,                   # 100
                 action_embedding_dim: int,                 # 200
                 gnn: bool = True,                          # True
                 decoder_use_graph_entities: bool = True,   # True
                 decoder_self_attend: bool = True,          # True
                 gnn_timesteps: int = 2,                    # 2
                 parse_sql_on_decoding: bool = True,        # True
                 add_action_bias: bool = False,              # True. Not define in jsonnet file.
                 use_neighbor_similarity_for_linking: bool = True, # True
                 dataset_path: str = 'dataset',             # dataset_path in jsonnet
                 training_beam_size: int = None,            # 1
                 decoder_num_layers: int = 1,               # 1. Not define in jsonnet file.
                 dropout: float = 0.0,                      # 0.5
                 rule_namespace: str = 'rule_labels',       # 'rule_labels'. Not define in jsonnet file.
                 scoring_dev_params: dict = None,           # None. Not define in jsonnet file.
                 debug_parsing: bool = False) -> None:      # False. Not define in jsonnet file.
        super().__init__(vocab)
        self.vocab = vocab          # I think it is automatically builded from Field of instance whose 'include_in_vocab' is True
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace
        self._question_embedder = question_embedder
        self._add_action_bias = add_action_bias
        self._scoring_dev_params = scoring_dev_params or {}
        self.parse_sql_on_decoding = parse_sql_on_decoding
        
        # TimeDistributed can automatically make entity_encoder support 4D tensor.
        # TimeDistributed will automatically reshapes the input of entity_encoder to a 3D tensor,
        # which is from 4D (batch,??,time_step,dimension) -> 3D (batch*??, time_step,dimension).
        # After calculation by entity_encoder, TimeDistributed reshapes it back to 4D.
        # _entity_encoder will calc the average of the embeding value of every token in one node.
        # For example: to "department id", the process of _entity_encoder is:
        # _e_e = [embeding(department)+embeding(id)]/2. The dimension of the _e_e is the same as the embeding(department).
        self._entity_encoder = TimeDistributed(entity_encoder)
        self._use_neighbor_similarity_for_linking = use_neighbor_similarity_for_linking
        self._self_attend = decoder_self_attend
        self._decoder_use_graph_entities = decoder_use_graph_entities

        self._action_padding_index = -1  # the padding value used by IndexField

        self._exact_match = Average()
        self._sql_evaluator_match = Average()
        self._action_similarity = Average()
        self._acc_single = Average()
        self._acc_multi = Average()
        self._beam_hit = Average()

        self._action_embedding_dim = action_embedding_dim

        num_actions = vocab.get_vocab_size(self._rule_namespace)
        if self._add_action_bias:
            input_action_dim = action_embedding_dim + 1
        else:
            input_action_dim = action_embedding_dim
        self._action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=input_action_dim)
        self._output_action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)

        encoder_output_dim = encoder.get_output_dim()
        if gnn:
            encoder_output_dim += action_embedding_dim

        # Initial value of these parameter is Random.
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self._first_attended_utterance = torch.nn.Parameter(torch.FloatTensor(encoder_output_dim))
        self._first_attended_output = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding)
        torch.nn.init.normal_(self._first_attended_utterance)
        torch.nn.init.normal_(self._first_attended_output)

        self._num_entity_types = 9 # ['boolean', 'foreign', 'number', 'others', 'primary', 'text', 'time'] + 'string' + 'table'
        self._embedding_dim = question_embedder.get_output_dim()

        self._entity_type_encoder_embedding = Embedding(self._num_entity_types, self._embedding_dim)
        self._entity_type_decoder_embedding = Embedding(self._num_entity_types, action_embedding_dim)

        self._linking_params = torch.nn.Linear(16, 1)
        torch.nn.init.uniform_(self._linking_params.weight, 0, 1)

        num_edge_types = 3
        if gnn:
            self._gnn = GatedGraphConv(self._embedding_dim, gnn_timesteps, num_edge_types=num_edge_types, dropout=dropout)
        else:
            self._gnn = None
        self._decoder_num_layers = decoder_num_layers

        self._beam_search = decoder_beam_search
        self._decoder_trainer = MaximumMarginalLikelihood(training_beam_size)

        if decoder_self_attend:
            self._transition_function = AttendPastSchemaItemsTransitionFunction(encoder_output_dim=encoder_output_dim, # 400+200gnn=600
                                                                                action_embedding_dim=action_embedding_dim, # 200
                                                                                input_attention=input_attention, # {"type": "dot_product"}
                                                                                past_attention=past_attention, # {"type": "dot_product"},
                                                                                predict_start_type_separately=False,
                                                                                add_action_bias=self._add_action_bias, # True
                                                                                dropout=dropout, # 0.5
                                                                                num_layers=self._decoder_num_layers) # 1
        else:
            self._transition_function = LinkingTransitionFunction(encoder_output_dim=encoder_output_dim,
                                                                  action_embedding_dim=action_embedding_dim,
                                                                  input_attention=input_attention,
                                                                  predict_start_type_separately=False,
                                                                  add_action_bias=self._add_action_bias,
                                                                  dropout=dropout,
                                                                  num_layers=self._decoder_num_layers)

        self._ent2ent_ff = FeedForward(action_embedding_dim, 1, action_embedding_dim, Activation.by_name('relu')())

        self._neighbor_params = torch.nn.Linear(self._embedding_dim, self._embedding_dim)

        # TODO: Remove hard-coded dirs
        self._evaluate_func = partial(evaluate,
                                      db_dir=os.path.join(dataset_path, 'database'),
                                      table=os.path.join(dataset_path, 'tables.json'),
                                      check_valid=False)

        self.debug_parsing = debug_parsing

    @overrides
    def forward(self,  # type: ignore
                utterance: Dict[str, torch.LongTensor],
                valid_actions: List[List[ProductionRule]],
                world: List[SpiderWorld],
                schema: Dict[str, torch.LongTensor],
                action_sequence: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        The five input parameter look like that it must the same as the "return Instance(fields)" in dataset_readers/spider.py
        
        utterance:
            It had been tokenized before return Instance(fields). But it will automatically be converted to single numbers by AllenNLP.
            (PS: tokenized is converting a sentence to a group word. Vocabulary is converting these word to numbers.)
            AllenNLP will build the vocabulary automatically from all words that is why 
            we can not get the numbers when debug in SpiderDatasetReader since we can build a vocabulary until we get all words.
        
        schema:
            This Dict is generated from SpiderKnowledgeGraphField obj which inherit from KnowledgeGraphField.
            This Dict will contain two elements: 'linking' and 'text'.

            1. 'text':
            The 'text' is 'entity_texts' in the graph which can consider as node name or node text. It is table name or column name in here.
            Because we do not include the words in 'text' to the vocabulary (The attribute of 'include_in_vocab' in SpiderKnowledgeGraphField is set False)
            So if we can not find the exact same words from the question dataset, we can only use @@UN_KNOWN@@ token for that.
            For example, suppose there are only two question(utterance) in the Spider dataset:
                Q1: how many heads of the departments are older than 56?
                Q2: show me all the student's ids.
                'text' in schema for Q1:[department, original department id, department head, staff, staff id, staff age]
                'text' in schema for Q2:[student, student id, student name]
                All the number for 'text' in schema for Q1 and Q2 will be @@UN_KNOWN@@.
                Since the @@UN_KNOWN@@ is 1, and @@PAD@@ is 0, we can get final token number for 'text' in schema is:
                'text' in schema for Q1:[[1,0,0],[1,1,1],[1,1,0],[1,0,0],[1,1,0],[1,1,0]]
                'text' in schema for Q2:[[1,0],[1,1],[1,1]]
                ----------------------
                But if the Q2 is: show me all the student's id.
                Now the vocabulary will contain the id and 'text' in schema for Q1 and Q2 also contain the token of 'id'.
                Let's suppose the number for 'id' in vocabulary is 5.Now:
                'text' in schema for Q1:[[1,0,0],[1,1,5],[1,1,0],[1,0,0],[1,5,0],[1,1,0]]
                'text' in schema for Q2:[[1,0],[1,5],[1,1]]

            2. 'linking'
            The linking come from linking_features. However, there dimension is different. 
            I think that is why: "self._feature_extractors = feature_extractors * 2" happen in SpiderKnowledgeGraphField.
            The shape of linking_features will be:
            [ <len of KnowledgeGraph.entities (graph node number)> * <len of utterance_tokens> * <len of feature_extractors> ]
            But the shape of 'linking' will be:
            [ <len of KnowledgeGraph.entities (graph node number)> * <len of utterance_tokens> * <len of feature_extractors * 2> ]
            Data that appears in front and has the same dimensions as linking_features is the same as linking_features.
            I don't know why it will add extra data to become <len of feature_extractors * 2>. 
            I also don't know what extra data means.
            But I think we can only use the data at the head whose shape is equal to linking_features, 
            which means we directly use the linking_features only.

        action_sequence:
            correct grammar for current sql.
        """
        batch_size = len(world)
        
        device = utterance['tokens'].device

        initial_state = self._get_initial_state(utterance, world, schema, valid_actions)

        if action_sequence is not None:
            # Remove the trailing dimension (from ListField[ListField[IndexField]]).
            action_sequence = action_sequence.squeeze(-1)
            action_seq_mask = action_sequence != self._action_padding_index
        else:
            action_seq_mask = None

        if self.training:
            # Use the _transition_function to controle the decoding process and decode the initial_state.
            # And then calc the loss between the decoding results and action_sequence.
            decode_output = self._decoder_trainer.decode(initial_state,
                                                         self._transition_function,
                                                         # supervision = (action_sequence.unsqueeze(1), action_seq_mask.unsqueeze(1)
                                                         (action_sequence.unsqueeze(1), action_seq_mask.unsqueeze(1))) 

            return {'loss': decode_output['loss']}
        else: # validation 
            loss = torch.tensor([0]).float().to(device)
            if action_sequence is not None and action_sequence.size(1) > 1:
                try:
                    loss = self._decoder_trainer.decode(initial_state,
                                                        self._transition_function,
                                                        (action_sequence.unsqueeze(1),
                                                         action_seq_mask.unsqueeze(1)))['loss']
                except ZeroDivisionError:
                    # reached a dead-end during beam search
                    pass

            outputs: Dict[str, Any] = {
                'loss': loss
            }

            num_steps = self._max_decoding_steps
            # This tells the state to start keeping track of debug info, which we'll pass along in
            # our output dictionary.
            initial_state.debug_info = [[] for _ in range(batch_size)]

            best_final_states = self._beam_search.search(num_steps,
                                                         initial_state,
                                                         self._transition_function,
                                                         keep_final_unfinished_states=False)

            self._compute_validation_outputs(valid_actions,
                                             best_final_states,
                                             world,
                                             action_sequence,
                                             outputs)
            return outputs

    def _get_initial_state(self,
                           utterance: Dict[str, torch.LongTensor],
                           worlds: List[SpiderWorld],
                           schema: Dict[str, torch.LongTensor],
                           actions: List[List[ProductionRule]]) -> GrammarBasedState:
        schema_text = schema['text']
        embedded_schema = self._question_embedder(schema_text, num_wrapping_dims=1)
        schema_mask = util.get_text_field_mask(schema_text, num_wrapping_dims=1).float()

        embedded_utterance = self._question_embedder(utterance)
        utterance_mask = util.get_text_field_mask(utterance).float()

        batch_size, num_entities, num_entity_tokens, _ = embedded_schema.size()
        num_entities = max([len(world.db_context.knowledge_graph.entities) for world in worlds]) # entities are nodes
        num_question_tokens = utterance['tokens'].size(1) # Because of pad, their size will be equal.

        # entity_types: tensor with shape (batch_size, num_entities), where each entry is the
        # entity's type id.
        # entity_type_dict: Dict[int, int], mapping flattened_entity_index -> type_index
        # These encode the same information, but for efficiency reasons later it's nice
        # to have one version as a tensor and one that's accessible on the cpu.
        entity_types, entity_type_dict = self._get_type_vector(worlds, num_entities, embedded_schema.device)

        entity_type_embeddings = self._entity_type_encoder_embedding(entity_types)

        # Compute entity and question word similarity.  We tried using cosine distance here, but
        # because this similarity is the main mechanism that the model can use to push apart logit
        # scores for certain actions (like "n -> 1" and "n -> -1"), this needs to have a larger
        # output range than [-1, 1].
        question_entity_similarity = torch.bmm(embedded_schema.view(batch_size,
                                                                    num_entities * num_entity_tokens,
                                                                    self._embedding_dim),
                                               torch.transpose(embedded_utterance, 1, 2))

        question_entity_similarity = question_entity_similarity.view(batch_size,
                                                                     num_entities,
                                                                     num_entity_tokens,
                                                                     num_question_tokens)
        # (batch_size, num_entities, num_question_tokens)
        question_entity_similarity_max_score, _ = torch.max(question_entity_similarity, 2)

        # (batch_size, num_entities, num_question_tokens, num_features)
        linking_features = schema['linking']

        linking_scores = question_entity_similarity_max_score

        # linking_features -> nn(16->1) -> feature_scores
        # Conbine the 16 linking features to 1
        feature_scores = self._linking_params(linking_features).squeeze(3)

        # linking_scores is created by embeding_of_utterance and embeding_of_node, which is trainable.
        # feature_scores is created by certain function, which is not trainable.
        # But they are the same. They means the similarity between utterance and node.
        linking_scores = linking_scores + feature_scores

        # (batch_size, num_question_tokens, num_entities)
        linking_probabilities = self._get_linking_probabilities(worlds, linking_scores.transpose(1, 2),
                                                                utterance_mask, entity_type_dict)

        # the shape of neighbor_indices is: (batch_size, num_entities, num_neighbors) or None
        # It is index that is generated from worlds[*].db_context.knowledge_graph.neighbors. It is the edge in a graph.
        # The @indices value@ here can be used to get the node and node value through: 
        # for b in batch...
        #   for n in num_entities
        #       for index in @indices value@[b][n]:
        #           node = worlds[i].db_context.knowledge_graph.entities[index] # node. such as: 'column:foreign:management:department_id'
        #           worlds[i].db_context.knowledge_graph.entity_text[node] # node value. such as: 'department id'
        neighbor_indices = self._get_neighbor_indices(worlds, num_entities, linking_scores.device)

        if self._use_neighbor_similarity_for_linking and neighbor_indices is not None:
            # (batch_size, num_entities, embedding_dim)
            encoded_table = self._entity_encoder(embedded_schema, schema_mask)

            # Neighbor_indices is padded with -1 since 0 is a potential neighbor index.
            # Thus, the absolute value needs to be taken in the index_select, and 1 needs to
            # be added for the mask since that method expects 0 for padding.
            # (batch_size, num_entities, num_neighbors, embedding_dim)
            embedded_neighbors = util.batched_index_select(encoded_table, torch.abs(neighbor_indices))

            
            neighbor_mask = util.get_text_field_mask({'ignored': neighbor_indices + 1},
                                                     num_wrapping_dims=1).float()

            # Encoder initialized to easily obtain a masked average.
            neighbor_encoder = TimeDistributed(BagOfEmbeddingsEncoder(self._embedding_dim, averaged=True))
            # (batch_size, num_entities, embedding_dim)
            embedded_neighbors = neighbor_encoder(embedded_neighbors, neighbor_mask)
            projected_neighbor_embeddings = self._neighbor_params(embedded_neighbors.float())

            # (batch_size, num_entities, embedding_dim)
            entity_embeddings = torch.tanh(entity_type_embeddings + projected_neighbor_embeddings)
        else:
            # (batch_size, num_entities, embedding_dim)
            entity_embeddings = torch.tanh(entity_type_embeddings)

        # The shape of entity_embeddings is (batch, node_num, dim)
        # The shape of linking_probabilities is (batch, utterance_len, node_num)
        # The shape of link_embedding is (batch, utterance_len, dim)
        # So actually, link_embedding = torch.bmm(linking_probabilities, entity_embeddings)
        # Actually, matrix multiply and attention is weighted_sum.
        # The link_embedding is embeding for every token in utterance with its node information.
        link_embedding = util.weighted_sum(entity_embeddings, linking_probabilities)

        # So we can catenate the link_embedding with embedded_utterance.
        encoder_input = torch.cat([link_embedding, embedded_utterance], 2)

        # (batch_size, utterance_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._encoder(encoder_input, utterance_mask))
        
        # linking_probabilities.max(dim=1)[1] is the index of the max value. So we only keep [0].
        # The shape of linking_probabilities is (batch, utterance_len, node_num)
        # So max_entities_relevance is the max relevant value for every node.
        max_entities_relevance = linking_probabilities.max(dim=1)[0]
        entities_relevance = max_entities_relevance.unsqueeze(-1).detach()

        # dot multiply. narrow the non-relevant node embeding value. Good!
        graph_initial_embedding = entity_type_embeddings * entities_relevance

        encoder_output_dim = self._encoder.get_output_dim()
        if self._gnn:

            # the shape of entities_graph_encoding is the same as graph_initial_embedding (batch, max_node_num, embed_dim)
            # Encode the graph_initial_embedding through the graph CNN
            entities_graph_encoding = self._get_schema_graph_encoding(worlds, graph_initial_embedding)

            # The same as previous
            # The shape of graph_link_embedding is (batch, max_utterance_num, embed_dim)
            # Now every word of token contain its node and the related node.
            graph_link_embedding = util.weighted_sum(entities_graph_encoding, linking_probabilities)
            encoder_outputs = torch.cat((
                encoder_outputs,
                graph_link_embedding
            ), dim=-1)

            encoder_output_dim = self._action_embedding_dim + self._encoder.get_output_dim()
        else:
            entities_graph_encoding = None

        if self._self_attend and entities_graph_encoding is not None:
            # linked_actions_linking_scores = self._get_linked_actions_linking_scores(actions, entities_graph_encoding)

            # _ent2ent_ff is just a activated function of relu
            # entities_graph_encoding -> relu -> entities_ff
            entities_ff = self._ent2ent_ff(entities_graph_encoding) # relue

            # Similarity between the every node.
            # The shape of linked_actions_linking_scores is (batch, max_node_num, max_node_num)
            # The first line:  [Similarity of Node1 and Node1, Similarity of Node1 and Node2, ... ]
            # The Second line: [Similarity of Node2 and Node1, Similarity of Node2 and Node2, ... ]
            # The true linked_actions_linking_scores is not related to the action. It is the self-connection of node.
            # The name is error.
            linked_actions_linking_scores = torch.bmm(entities_ff, entities_ff.transpose(1, 2))
        else:
            linked_actions_linking_scores = [None] * batch_size

        # This will be our initial hidden state and memory cell for the decoder LSTM.
        # encoder_outputs contain three information: utterance, utterance related node, utterance related node graph
        # There is no RNN in get_final_encoder_states. We only take the hidden state from encoder_outputs.
        # The shape of final_encoder_output is (batch, dim_of_encoder_outputs)
        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             utterance_mask,
                                                             self._encoder.is_bidirectional())


        memory_cell = encoder_outputs.new_zeros(batch_size, encoder_output_dim)
        initial_score = embedded_utterance.data.new_zeros(batch_size)

        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, utterance_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(utterance_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        encoder_output_list = [encoder_outputs[i] for i in range(batch_size)]
        utterance_mask_list = [utterance_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(  RnnStatelet(     # RnnStatelet is for decoder
                                                 final_encoder_output[i],       # The final hidden state output from encoder and it will be the initial hidden_state for decoder RNN(LSTM)
                                                 memory_cell[i],                # Here is 0. Just for LSTM. But this project run based on LSTM.
                                                 self._first_action_embedding,  # previous_action_embedding. Randon normal_ init.
                                                 self._first_attended_utterance,# attended_input. Randon normal_ init.
                                                 encoder_output_list,           # encoder_output
                                                 utterance_mask_list)           # encoder_output_mask
                                    )

        initial_grammar_state = [self._create_grammar_state(worlds[i],
                                                            actions[i],
                                                            linking_scores[i], # the similarity between utterance and node. shape: (node_num, utterance_num)
                                                            linked_actions_linking_scores[i],
                                                            entity_types[i],
                                                            entities_graph_encoding[
                                                                i] if entities_graph_encoding is not None else None)
                                 for i in range(batch_size)]

        # self.parse_sql_on_decoding is True
        initial_sql_state = [SqlState(actions[i], self.parse_sql_on_decoding) for i in range(batch_size)]

        # include all batch.
        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_state,
                                          sql_state=initial_sql_state,
                                          possible_actions=actions,
                                          action_entity_mapping=[w.get_action_entity_mapping() for w in worlds])

        return initial_state



    @staticmethod
    def _get_neighbor_indices(worlds: List[SpiderWorld],
                              num_entities: int,
                              device: torch.device) -> torch.LongTensor:
        """
        This method returns the indices of each entity's neighbors. A tensor
        is accepted as a parameter for copying purposes.

        Parameters
        ----------
        worlds : ``List[SpiderWorld]``
        num_entities : ``int``
        tensor : ``torch.Tensor``
            Used for copying the constructed list onto the right device.

        Returns
        -------
        A ``torch.LongTensor`` with shape ``(batch_size, num_entities, num_neighbors)``. It is padded
        with -1 instead of 0, since 0 is a valid neighbor index. If all the entities in the batch
        have no neighbors, None will be returned.
        """

        num_neighbors = 0
        for world in worlds:
            for entity in world.db_context.knowledge_graph.entities:
                if len(world.db_context.knowledge_graph.neighbors[entity]) > num_neighbors:
                    num_neighbors = len(world.db_context.knowledge_graph.neighbors[entity])

        batch_neighbors = []
        no_entities_have_neighbors = True
        for world in worlds:
            # Each batch instance has its own world, which has a corresponding table.
            entities = world.db_context.knowledge_graph.entities
            entity2index = {entity: i for i, entity in enumerate(entities)}
            entity2neighbors = world.db_context.knowledge_graph.neighbors
            neighbor_indexes = []
            for entity in entities:
                entity_neighbors = [entity2index[n] for n in entity2neighbors[entity]]
                if entity_neighbors:
                    no_entities_have_neighbors = False
                # Pad with -1 instead of 0, since 0 represents a neighbor index.
                padded = pad_sequence_to_length(entity_neighbors, num_neighbors, lambda: -1)
                neighbor_indexes.append(padded)
            neighbor_indexes = pad_sequence_to_length(neighbor_indexes,
                                                      num_entities,
                                                      lambda: [-1] * num_neighbors)
            batch_neighbors.append(neighbor_indexes)
        # It is possible that none of the entities has any neighbors, since our definition of the
        # knowledge graph allows it when no entities or numbers were extracted from the question.
        if no_entities_have_neighbors:
            return None
        return torch.tensor(batch_neighbors, device=device, dtype=torch.long)



    def _get_schema_graph_encoding(self,
                                   worlds: List[SpiderWorld],
                                   initial_graph_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_num_entities = max([len(world.db_context.knowledge_graph.entities) for world in worlds])
        batch_size = initial_graph_embeddings.size(0)

        graph_data_list = []

        for batch_index, world in enumerate(worlds):
            x = initial_graph_embeddings[batch_index]
            
            # edge list
            adj_list = self._get_graph_adj_lists(initial_graph_embeddings.device,
                                                 world, initial_graph_embeddings.size(1) - 1)
            graph_data = Data(x)
            for i, l in enumerate(adj_list):
                graph_data[f'edge_index_{i}'] = l
            # After the for loop, we can:
            # graph_data.x is node
            # graph_data.edge_index_0 is adj_list[0]
            # graph_data.edge_index_1 is adj_list[1]
            # graph_data.edge_index_2 is adj_list[2]

            graph_data_list.append(graph_data)


        batch = Batch.from_data_list(graph_data_list)

        gnn_output = self._gnn(batch.x, [batch[f'edge_index_{i}'] for i in range(self._gnn.num_edge_types)])

        num_nodes = max_num_entities
        gnn_output = gnn_output.view(batch_size, num_nodes, -1)

        # entities_encodings = gnn_output
        entities_encodings = gnn_output[:, :max_num_entities] # I think this is useless. # I prefer the last command. So I add:

        assert entities_encodings.shape[0] == gnn_output.shape[0]
        assert entities_encodings.shape[1] == gnn_output.shape[1]
        assert entities_encodings.shape[2] == gnn_output.shape[2]
        assert entities_encodings.shape[2]*entities_encodings.shape[1]*entities_encodings.shape[0] == torch.sum( torch.eq(entities_encodings.cpu().detach(), gnn_output.cpu().detach()) )

        return entities_encodings

    @staticmethod
    def _get_graph_adj_lists(device, world, global_entity_id, global_node=False):
        """
        get the edge of node.
        return a list contain 3 obj:
           ( 
            [
                edge: "column to table" and "table to column". For example, if there is [1,5], it must also contain [5,1]. 
            ],
            [
                foreign->primary
            ],
            [
                foreign<-primary. For example, if there is [2,5] in last "foreign->primary", there must be a [5,2] in here.
            ]
           )
        """
        entity_mapping = {}
        for i, entity in enumerate(world.db_context.knowledge_graph.entities):
            entity_mapping[entity] = i
        entity_mapping['_global_'] = global_entity_id
        adj_list_own = []  # column--table (bi-direction)
        adj_list_link = []  # table->table / foreign->primary
        adj_list_linked = []  # table<-table / foreign<-primary
        adj_list_global = []  # node->global

        # TODO: Prepare in advance?
        for key, neighbors in world.db_context.knowledge_graph.neighbors.items():
            idx_source = entity_mapping[key]
            for n_key in neighbors:
                idx_target = entity_mapping[n_key]
                if n_key.startswith("table") or key.startswith("table"):
                    adj_list_own.append((idx_source, idx_target))
                elif n_key.startswith("string") or key.startswith("string"):
                    adj_list_own.append((idx_source, idx_target))
                elif key.startswith("column:foreign"):
                    adj_list_link.append((idx_source, idx_target))
                    src_table_key = f"table:{key.split(':')[2]}"
                    tgt_table_key = f"table:{n_key.split(':')[2]}"
                    idx_source_table = entity_mapping[src_table_key]
                    idx_target_table = entity_mapping[tgt_table_key]
                    adj_list_link.append((idx_source_table, idx_target_table))
                elif n_key.startswith("column:foreign"):
                    adj_list_linked.append((idx_source, idx_target))
                    src_table_key = f"table:{key.split(':')[2]}"
                    tgt_table_key = f"table:{n_key.split(':')[2]}"
                    idx_source_table = entity_mapping[src_table_key]
                    idx_target_table = entity_mapping[tgt_table_key]
                    adj_list_linked.append((idx_source_table, idx_target_table))
                else:
                    assert False

            adj_list_global.append((idx_source, entity_mapping['_global_']))

        all_adj_types = [adj_list_own, adj_list_link, adj_list_linked]

        if global_node:
            all_adj_types.append(adj_list_global)

        return [torch.tensor(l, device=device, dtype=torch.long).transpose(0, 1) if l
                else torch.tensor(l, device=device, dtype=torch.long)
                for l in all_adj_types]


    def _create_grammar_state(self,
                              world: SpiderWorld,
                              possible_actions: List[ProductionRule],
                              linking_scores: torch.Tensor,
                              self_linking_scores: torch.Tensor,
                              entity_types: torch.Tensor,
                              entity_graph_encoding: torch.Tensor) -> GrammarStatelet:
        """
        This function create a GrammarStatelet obj.
        Every SQL correspond to one GrammarStatelet. One GrammarBasedState (can) correspond to one batch data.
        This function generate a GrammarStatelet obj based on the dict-type-action (see dict-type-action in spider.py).
        And then translate the string value-list to tensor. 
        Value-list is possible action in global or possible column/table in linked.
        (String value-list -> index (through a vocabulary) -> embeding -> tensor )
        But this process only support the global rules. So we will get different data between global and non-global.
        Here, non-global is 'linked'.
        Notice: There will not be 'global' and 'linked' appearing in one key although the program support this situation but the real world does not contain this.
        So we can get the translated_valid_actions which base on dict-type-action and is the major information in GrammarStatelet.
        translated_valid_actions:
        {       #key_1       #key_2    #value-list
               arg_list:{
                            global:(
                                        tensor_1. # embeding the value-list. Using action_embedder.
                                            # Its shape: (number_of_value-list, embeding_dim_1). (2,201) or (2,200)
                                            # number_of_value-list is 2 in 'arg_list', because there are two value-list for the key of 'arg_list'. (check it in spider.py).
                                            # if self._add_action_bias is True, the embeding_dim_1 = 201; else, embeding_dim_1 = 200.

                                        tensor_2. # embeding the value-list. Using output_action_embedder.
                                            # Its shape: (number_of_value-list, embeding_dim_2). (2,200)
                                            # self._add_action_bias makes no effect here. embeding_dim_2 always = 200.

                                        list_1. # List the index of the value-list. I do not know whether it is the index in the vocabulary.
                                            # But I am sure we can found the exact action through the index and the list-type-action. (see list-type-action in spider.py)
                                            # As the same, the len(list_1) equal to number_of_value-list (2).
                            )
                        }
                column_name:{
                            linked:(
                                        tensor_1. # entity_link_utterance_scores
                                            # Its shape is (number_of_value-list, number_utterance_token)
                                            # number_of_value-list is the number of columns in all tables in the database for the key 'column_name'.
                                            # It represent the similarity between columns and the utterance.

                                        tensor_2. # entity_graph_embeddings
                                            # Shape: (number_of_value-list, dim_of_entity_graph_embeddings). (number_of_value-list, 200)
                                            # Notice: number_of_value-list do not equal to the number of the entities (nodes).
                                            #         entities (nodes) include the column (name) and the table (table) in this code.
                                        
                                        list_1.   # List the index of the value-list for 'column_name'. I think these indices have no relationship with the vocabulary.
                                            # I am sure we can found the exact action through the index and the list-type-action. (see list-type-action in spider.py)
                                            # As the same, the len(list_1) equal to number_of_value-list.

                                        tensor_3. # entity_self_linking_scores
                                            # It will appear when self._self_attend is True.
                                            # Shape: (number_of_value-list, number_of_entities).
                                            # (number_of_entities - number_of_value-list) equal to the number_of_table here, 
                                            # because the number_of_value-list equal to number_of_column for the key 'column_name'. 
                                            # And the entities (nodes) always include the column (name) and the table (table).
                                            # So it means the similarity between the column_name and entities (nodes).
                                            # Since the entities (nodes) include the column_name information, so it call self attention.
                            )
                }
                           
               ...: {....}
               ... 
        }
        
        possible_actions:
            The total actions can be appeared in the current SQL. The detailed of this obj, please see the end of the spider.py.
        linking_scores:
            Node linking to utterance.
            the similarity between utterance and node. shape: (node_num, utterance_num)
        self_linking_scores:
            Node self linking.
            The true self_linking_scores is not related to the action. It is the self-connection of node.
            The shape of self_linking_scores is (batch, max_node_num, max_node_num)
            The first line:  [Similarity of Node1 and Node1, Similarity of Node1 and Node2, ... ]
            The Second line: [Similarity of Node2 and Node1, Similarity of Node2 and Node2, ... ]
        """
        action_map = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action_map[action_string] = action_index

        valid_actions = world.valid_actions
        entity_map = {}
        entities = world.entities_names

        for entity_index, entity in enumerate(entities):
            entity_map[entity] = entity_index

        translated_valid_actions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]] = {}
        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.  We'll first split those productions by global vs.
            # linked action.

            action_indices = [action_map[action_string] for action_string in action_strings] #index in possible_actions
            production_rule_arrays = [(possible_actions[index], index) for index in action_indices]
            global_actions = []
            linked_actions = []
            for production_rule_array, action_index in production_rule_arrays:
                if production_rule_array[1]: 
                    # this rule comes from the global grammar (global rules) see the end of spider.py.
                    # production_rule_array[2] is _rule_id built by allennlp. 
                    # _rule_id will be the same in all instance if the rule is the same. Similar to the vocabulary machenism.
                    global_actions.append((production_rule_array[2], action_index))  # keep rule id
                else:
                    # this rule is an instance-specific production rule. not global rules.
                    linked_actions.append((production_rule_array[0], action_index))  # keep rule string value

            if global_actions:
                global_action_tensors, global_action_ids = zip(*global_actions)
                global_action_tensor = torch.cat(global_action_tensors, dim=0).to(
                    global_action_tensors[0].device).long() # cat the dim=0, and then to GPU, and then to long tensor.

                # global_input_embeddings is used for calc the similarity with the predicted_action_embedding.
                # We can choose the higher similarity one as the correct prediction.
                # global_output_embeddings is used for the make_state function in AttendPastSchemaItemsTransitionFunction or BasicTransitionFunction.
                # It will become the new RnnStatelet.previous_action_embedding for tje mew state.
                global_input_embeddings = self._action_embedder(global_action_tensor)
                global_output_embeddings = self._output_action_embedder(global_action_tensor)
                translated_valid_actions[key]['global'] = (global_input_embeddings,
                                                           global_output_embeddings,
                                                           list(global_action_ids))

            if linked_actions:
                linked_rules, linked_action_ids = zip(*linked_actions)
                entities = [rule.split(' -> ')[1].strip('[]\"') for rule in linked_rules]

                entity_ids = [entity_map[entity] for entity in entities]

                # here means the order of the 'linking_scores' is the same as the 'world'
                # So we can get our linking_scores through this way:
                entity_link_utterance_scores = linking_scores[entity_ids] # I change the name: # entity_linking_scores = linking_scores[entity_ids]
                if self_linking_scores is not None:
                    entity_self_linking_scores  = self_linking_scores[entity_ids] # I change the name: # entity_action_linking_scores = self_linking_scores[entity_ids]

                if not self._decoder_use_graph_entities:
                    entity_type_tensor = entity_types[entity_ids]
                    entity_type_embeddings = (self._entity_type_decoder_embedding(entity_type_tensor)
                                              .to(entity_types.device)
                                              .float())
                else:
                    entity_type_embeddings = entity_graph_encoding.index_select(
                        dim=0,
                        index=torch.tensor(entity_ids, device=entity_graph_encoding.device)
                    )

                if self._self_attend:
                    translated_valid_actions[key]['linked'] = (entity_link_utterance_scores, # I change the name: # entity_linking_scores,
                                                               entity_type_embeddings,       # It should be 'entity_graph_embeddings' in here.
                                                               list(linked_action_ids),
                                                               entity_self_linking_scores)   # I change the name: # entity_action_linking_scores)
                else:
                    translated_valid_actions[key]['linked'] = (entity_link_utterance_scores, # I change the name: # entity_linking_scores,
                                                               entity_type_embeddings,
                                                               list(linked_action_ids))

        # The translated_valid_actions is 
        return GrammarStatelet(['statement'],       # This is the first (start) symbol in SQL grammar (action) which is: 'statement -> [query, iue, query]' and 'statement -> [query]'.
                               translated_valid_actions,
                               self.is_nonterminal) # self.is_nonterminal is a callable function for GrammarStatelet

    @staticmethod
    def is_nonterminal(token: str):
        if token[0] == '"' and token[-1] == '"':
            return False
        return True

    def _get_linking_probabilities(self,
                                   worlds: List[SpiderWorld],
                                   linking_scores: torch.FloatTensor,
                                   question_mask: torch.LongTensor,
                                   entity_type_dict: Dict[int, int]) -> torch.FloatTensor:
        """
        Produces the probability of an entity given a question word and type. The logic below
        separates the entities by type since the softmax normalization term sums over entities
        of a single type.

        Parameters
        ----------
        worlds : ``List[WikiTablesWorld]``
        linking_scores : ``torch.FloatTensor``
            Has shape (batch_size, num_question_tokens, num_entities).
        question_mask: ``torch.LongTensor``
            Has shape (batch_size, num_question_tokens).
        entity_type_dict : ``Dict[int, int]``
            This is a mapping from ((batch_index * num_entities) + entity_index) to entity type id.

        Returns
        -------
        batch_probabilities : ``torch.FloatTensor``
            Has shape ``(batch_size, num_question_tokens, num_entities)``.
            Contains all the probabilities for an entity given a question word.
        """
        _, num_question_tokens, num_entities = linking_scores.size()
        batch_probabilities = []

        for batch_index, world in enumerate(worlds):
            all_probabilities = []
            num_entities_in_instance = 0

            # NOTE: The way that we're doing this here relies on the fact that entities are
            # implicitly sorted by their types when we sort them by name, and that numbers come
            # before "date_column:", followed by "number_column:", "string:", and "string_column:".
            # This is not a great assumption, and could easily break later, but it should work for now.
            for type_index in range(self._num_entity_types):
                # This index of 0 is for the null entity for each type, representing the case where a
                # word doesn't link to any entity.
                entity_indices = [0]
                entities = world.db_context.knowledge_graph.entities # there is no pad data in entities, so don't worry.
                for entity_index, _ in enumerate(entities):
                    if entity_type_dict[batch_index * num_entities + entity_index] == type_index:
                        entity_indices.append(entity_index)

                if len(entity_indices) == 1:
                    # No entities of this type; move along...
                    continue

                # We're subtracting one here because of the null entity we added above.
                num_entities_in_instance += len(entity_indices) - 1

                # We separate the scores by type, since normalization is done per type.  There's an
                # extra "null" entity per type, also, so we have `num_entities_per_type + 1`.  We're
                # selecting from a (num_question_tokens, num_entities) linking tensor on _dimension 1_,
                # so we get back something of shape (num_question_tokens,) for each index we're
                # selecting.  All of the selected indices together then make a tensor of shape
                # (num_question_tokens, num_entities_per_type + 1).
                # But Gan can not find that normalization is done per type.???
                indices = linking_scores.new_tensor(entity_indices, dtype=torch.long)
                entity_scores = linking_scores[batch_index].index_select(1, indices)

                # We used index 0 for the null entity, so this will actually have some values in it.
                # But we want the null entity's score to be 0, so we set that here.
                entity_scores[:, 0] = 0

                # No need for a mask here, as this is done per batch instance, with no padding.
                type_probabilities = torch.nn.functional.softmax(entity_scores, dim=1)
                all_probabilities.append(type_probabilities[:, 1:])

            # We need to add padding here if we don't have the right number of entities.
            if num_entities_in_instance != num_entities:
                zeros = linking_scores.new_zeros(num_question_tokens,
                                                 num_entities - num_entities_in_instance)
                all_probabilities.append(zeros)

            # (num_question_tokens, num_entities)
            probabilities = torch.cat(all_probabilities, dim=1)
            batch_probabilities.append(probabilities)
        batch_probabilities = torch.stack(batch_probabilities, dim=0)
        return batch_probabilities * question_mask.unsqueeze(-1).float()

    @staticmethod
    def _action_history_match(predicted: List[int], targets: torch.LongTensor) -> int:
        # TODO(mattg): this could probably be moved into a FullSequenceMatch metric, or something.
        # Check if target is big enough to cover prediction (including start/end symbols)
        if len(predicted) > targets.size(0):
            return 0
        predicted_tensor = targets.new_tensor(predicted)
        targets_trimmed = targets[:len(predicted)]
        # Return 1 if the predicted sequence is anywhere in the list of targets.
        return torch.max(torch.min(targets_trimmed.eq(predicted_tensor), dim=0)[0]).item()

    @staticmethod
    def _query_difficulty(targets: torch.LongTensor, action_mapping, batch_index):
        number_tables = len([action_mapping[(batch_index, int(a))] for a in targets if
                             a >= 0 and action_mapping[(batch_index, int(a))].startswith('table_name')])
        return number_tables > 1

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            '_match/exact_match': self._exact_match.get_metric(reset),
            'sql_match': self._sql_evaluator_match.get_metric(reset),
            '_others/action_similarity': self._action_similarity.get_metric(reset),
            '_match/match_single': self._acc_single.get_metric(reset),
            '_match/match_hard': self._acc_multi.get_metric(reset),
            'beam_hit': self._beam_hit.get_metric(reset)
        }

    @staticmethod
    def _get_type_vector(worlds: List[SpiderWorld],
                         num_entities: int,
                         device) -> Tuple[torch.LongTensor, Dict[int, int]]:
        """
        Produces the encoding for each entity's type. In addition, a map from a flattened entity
        index to type is returned to combine entity type operations into one method.

        Parameters
        ----------
        worlds : ``List[AtisWorld]``
        num_entities : ``int``
        tensor : ``torch.Tensor``
            Used for copying the constructed list onto the right device.

        Returns
        -------
        A ``torch.LongTensor`` with shape ``(batch_size, num_entities)``.
            It give every entities (nodes) a type and pad them alignment. So you can put it to embeding or NN.
            This tensor is also the entity_types but use different data organization. 
            In here, the location of entity type is the same as the SpiderWorld.db_context.knowledge_graph.entities.

        entity_types : ``Dict[int, int]``
            This is a mapping from ((batch_index * num_entities) + entity_index) to entity type id.
            This is almost the same as last return obj which is a tensor, but here is a dict.


        So their common:
            A ``torch.LongTensor``[3][1] is the type of SpiderWorld[3].db_context.knowledge_graph.entities[1].
            A ``torch.LongTensor``[3][1] equal to entity_types[3*num_entities+1]
        their different:
            Because of pad, total element of A ``torch.LongTensor`` may be more than entity_types. But the function <(batch_index * num_entities) + entity_index> is always correct.
            A ``torch.LongTensor is a tensor. entity_types is a dict.
        """
        entity_types = {}
        batch_types = []

        column_type_ids = ['boolean', 'foreign', 'number', 'others', 'primary', 'text', 'time']

        for batch_index, world in enumerate(worlds):
            types = []

            for entity_index, entity in enumerate(world.db_context.knowledge_graph.entities):
                parts = entity.split(':')
                entity_main_type = parts[0]

                if entity_main_type == 'column':
                    entity_type = column_type_ids.index(parts[1])
                elif entity_main_type == 'string':
                    # cell value
                    entity_type = len(column_type_ids)
                elif entity_main_type == 'table':
                    entity_type = len(column_type_ids) + 1
                else:
                    raise (Exception("Unkown entity"))
                types.append(entity_type)

                # For easier lookups later, we're actually using a _flattened_ version
                # of (batch_index, entity_index) for the key, because this is how the
                # linking scores are stored.
                flattened_entity_index = batch_index * num_entities + entity_index
                entity_types[flattened_entity_index] = entity_type
            padded = pad_sequence_to_length(types, num_entities, lambda: 0) # pad 0 for alignment. But 0 relate to the type of boolean. I think it is fine?! May be it will delete it later through mask.
            batch_types.append(padded)

        return torch.tensor(batch_types, dtype=torch.long, device=device), entity_types



    def _compute_validation_outputs(self,
                                    actions: List[List[ProductionRuleArray]],
                                    best_final_states: Mapping[int, Sequence[GrammarBasedState]],
                                    world: List[SpiderWorld],
                                    target_list: List[List[str]],
                                    outputs: Dict[str, Any]) -> None:
        """
        actions:
            all possible actions
        best_final_states:
            best 10(beam size) state
        world:
            SpiderWorld obj for this sql
        target_list:
            correct action index list. The index point to the actions obj.
        outputs:
            update this obj for function output
        """
        batch_size = len(actions)

        outputs['predicted_sql_query'] = []

        action_mapping = {} # action_mapping[(batch_index,action_index)] can get the action value
        for batch_index, batch_actions in enumerate(actions):
            for action_index, action in enumerate(batch_actions):
                action_mapping[(batch_index, action_index)] = action[0]

        for i in range(batch_size):
            # gold sql exactly as given
            original_gold_sql_query = ' '.join(world[i].get_query_without_table_hints())

            if i not in best_final_states:
                self._exact_match(0)
                self._action_similarity(0)
                self._sql_evaluator_match(0)
                self._acc_multi(0)
                self._acc_single(0)
                outputs['predicted_sql_query'].append('')
                continue

            best_action_indices = best_final_states[i][0].action_history[0]

            action_strings = [action_mapping[(i, action_index)]
                              for action_index in best_action_indices]
            predicted_sql_query = action_sequence_to_sql(action_strings, add_table_names=True)
            outputs['predicted_sql_query'].append(sqlparse.format(predicted_sql_query, reindent=False))

            if target_list is not None:
                targets = target_list[i].data

                sequence_in_targets = self._action_history_match(best_action_indices, targets)
                self._exact_match(sequence_in_targets)

                sql_evaluator_match = self._evaluate_func(original_gold_sql_query, predicted_sql_query, world[i].db_id)
                self._sql_evaluator_match(sql_evaluator_match)

                similarity = difflib.SequenceMatcher(None, best_action_indices, targets)
                self._action_similarity(similarity.ratio())

                difficulty = self._query_difficulty(targets, action_mapping, i)
                if difficulty:
                    self._acc_multi(sql_evaluator_match)
                else:
                    self._acc_single(sql_evaluator_match)

            beam_hit = False
            beam_sql_query = ""
            outputs['beam_sql_query'] = []
            for pos, final_state in enumerate(best_final_states[i]):
                action_indices = final_state.action_history[0]
                action_strings = [action_mapping[(i, action_index)]
                                  for action_index in action_indices]
                candidate_sql_query = action_sequence_to_sql(action_strings, add_table_names=True)

                if candidate_sql_query:
                    beam_sql_query += ":" + sqlparse.format(candidate_sql_query, reindent=False) + "\t" + str(final_state.score[0].detach().cpu().numpy()) + "\n"
                
                if target_list is not None:
                    correct = self._evaluate_func(original_gold_sql_query, candidate_sql_query, world[i].db_id)

                    if correct:
                        beam_hit = True
                    self._beam_hit(beam_hit)
            outputs['beam_sql_query'].append(beam_sql_query)
