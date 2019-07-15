from collections import defaultdict
from typing import Dict, Tuple, List, Set, Any, Callable, Optional

import torch
from allennlp.modules import Attention, FeedForward
from allennlp.nn import Activation, util

from allennlp.state_machines.states.grammar_based_state import GrammarBasedState
from allennlp.state_machines.transition_functions import BasicTransitionFunction
from allennlp.state_machines.transition_functions.linking_transition_function import LinkingTransitionFunction
from overrides import overrides
from torch.nn import Linear

from state_machines.states.rnn_statelet import RnnStatelet


class AttendPastSchemaItemsTransitionFunction(BasicTransitionFunction):
    def __init__(self,
                 encoder_output_dim: int,                               # 400+200gnn=600
                 action_embedding_dim: int,                             # 200
                 input_attention: Attention,                            # {"type": "dot_product"}
                 past_attention: Attention,                             # {"type": "dot_product"}
                 activation: Activation = Activation.by_name('relu')(), 
                 predict_start_type_separately: bool = True,            # False
                 num_start_types: int = None,
                 add_action_bias: bool = True,                          # True
                 dropout: float = 0.0,                                  # 0.5
                 num_layers: int = 1) -> None:                          # 1
                 
        super().__init__(encoder_output_dim=encoder_output_dim,
                         action_embedding_dim=action_embedding_dim,
                         input_attention=input_attention,
                         num_start_types=num_start_types,
                         activation=activation,
                         predict_start_type_separately=predict_start_type_separately,
                         add_action_bias=add_action_bias,
                         dropout=dropout,
                         num_layers=num_layers)

        self._past_attention = past_attention
        self._ent2ent_ff = FeedForward(1, 1, 1, Activation.by_name('linear')())



    @overrides
    def take_step(self,
                  state: GrammarBasedState,
                  max_actions: int = None,
                  allowed_actions: List[Set[int]] = None) -> List[GrammarBasedState]:

        """
        Automatically called by the beam search.

        state:
            all state include all batch. have already been combined.
        max_actions:
            beam_size.
        allowed_actions:
            There are too many names to make people feel confusion. 
            allowed_actions come from the action_sequence in SpiderParser.forward() and also from supervision in MaximumMarginalLikelihood.decode()
            There are almost the same but using different names. Bad namespace.
            So, we should use the allowed_actions to calc the loss.
            It is the correct answer including the actions (grammar) and SQL. SQL can be generated from the action. 
            Because the actions (grammar) include the specific (table,column name) and global rules.
            From it, we can understand the problem here is how to learn to generate the correct actions (grammar).

            Notice: allowed_actions is only the allowed actions for next action. It is part of the action_sequence.
                    allowed_actions is [ {action_index_1, ...} , {...} ]
        """

        # _predict_start_type_separately equal to predict_start_type_separately. It is always False in training.
        if self._predict_start_type_separately and not state.action_history[0]:
            # The wikitables parser did something different when predicting the start type, which
            # is our first action.  So in this case we break out into a different function.  We'll
            # ignore max_actions on our first step, assuming there aren't that many start types.
            return self._take_first_step(state, allowed_actions)

        # Taking a step in the decoder consists of three main parts.  First, we'll construct the
        # input to the decoder and update the decoder's hidden state.  Second, we'll use this new
        # hidden state (and maybe other information) to predict an action.  Finally, we will
        # construct new states for the next step.  Each new state corresponds to one valid action
        # that can be taken from the current state, and they are ordered by their probability of
        # being selected.

        updated_state = self._update_decoder_state(state)
        batch_results = self._compute_action_probabilities(state,
                                                           updated_state['hidden_state'],
                                                           updated_state['attention_weights'],
                                                           updated_state['past_schema_items_attention_weights'],
                                                           updated_state['predicted_action_embeddings'])
        new_states = self._construct_next_states(state,
                                                 updated_state,
                                                 batch_results,
                                                 max_actions,
                                                 allowed_actions) # correct answer include

        return new_states



    def _update_decoder_state(self, state: GrammarBasedState) -> Dict[str, torch.Tensor]:
        # For updating the decoder, we're doing a bunch of tensor operations that can be batched
        # without much difficulty.  So, we take all group elements and batch their tensors together
        # before doing these decoder operations.

        group_size = len(state.batch_indices) # batch size.
        attended_question = torch.stack([rnn_state.attended_input for rnn_state in state.rnn_state]) # state.rnn_state is RnnStatelet
        
        # conbine the hidden_state and memory_cell
        if self._num_layers > 1:
            hidden_state = torch.stack([rnn_state.hidden_state for rnn_state in state.rnn_state], 1)
            memory_cell = torch.stack([rnn_state.memory_cell for rnn_state in state.rnn_state], 1)
        else:
            hidden_state = torch.stack([rnn_state.hidden_state for rnn_state in state.rnn_state])
            memory_cell = torch.stack([rnn_state.memory_cell for rnn_state in state.rnn_state])

        # Initial value is Random because there is no previous_action.
        previous_action_embedding = torch.stack([rnn_state.previous_action_embedding
                                                 for rnn_state in state.rnn_state])

        # (group_size, decoder_input_dim)
        # self._input_projection_layer = Linear(output_dim + action_embedding_dim, encoder_output_dim) 600 -> 600
        # This code is wrote in BasicTransitionFunction.py
        projected_input = self._input_projection_layer(torch.cat([attended_question,
                                                                  previous_action_embedding], -1))
        decoder_input = self._activation(projected_input)

        # (attended_question + previous_action_embedding) -> LSTM decoder
        if self._num_layers > 1:
            # self._decoder_cell is a self._num_layers layers LSTM with encoder_output_dim -> encoder_output_dim
            _, (hidden_state, memory_cell) = self._decoder_cell(decoder_input.unsqueeze(0), # 
                                                                (hidden_state, memory_cell))
        else:
            # self._decoder_cell is a one layer LSTM with encoder_output_dim -> encoder_output_dim
            # Initial memory_cell is all 0.
            # Initial hidden_state is the final hidden state output from encoder.
            hidden_state, memory_cell = self._decoder_cell(decoder_input, (hidden_state, memory_cell))

        hidden_state = self._dropout(hidden_state)

        # (group_size, encoder_output_dim)
        # encoder_outputs are total hidden state from encoder.
        encoder_outputs = torch.stack([state.rnn_state[0].encoder_outputs[i] for i in state.batch_indices])
        encoder_output_mask = torch.stack([state.rnn_state[0].encoder_output_mask[i] for i in state.batch_indices])

        if self._num_layers > 1:
            attended_question, attention_weights = self.attend_on_question(hidden_state[-1],
                                                                           encoder_outputs,
                                                                           encoder_output_mask)
            # action_query conbine a lot of information not just the action.
            action_query = torch.cat([hidden_state[-1], attended_question], dim=-1)
        else:
            # attend_on_question is:
            # re = encoder_outputs.bmm(hidden_state.unsqueeze(-1)).squeeze(-1)
            # remove redundant re based on the mask
            # re -> softmax. Normalize the probability. And then we can get the attention_weights.
            # attended_question = util.weighted_sum(encoder_outputs, attention_weights)
            attended_question, attention_weights = self.attend_on_question(hidden_state,
                                                                           encoder_outputs,
                                                                           encoder_output_mask)
            # action_query conbine a lot of information not just the action.                                                               
            action_query = torch.cat([hidden_state, attended_question], dim=-1)

        # TODO: Can batch this (need to save ids of states with saved outputs)
        past_schema_items_attention_weights = []
        for i, rnn_state in enumerate(state.rnn_state):
            if rnn_state.decoder_outputs is not None:   # Initial is None because there is not decoding before running this decoder
                decoder_outputs_states, decoder_outputs_ids = rnn_state.decoder_outputs
                attn_weights = self.attend(self._past_attention, hidden_state[i].unsqueeze(0),
                                           decoder_outputs_states.unsqueeze(0), None).squeeze(0)
                past_schema_items_attention_weights.append((attn_weights, decoder_outputs_ids))
            else:
                past_schema_items_attention_weights.append(None)

        # past_schema_items_attention_weights = torch.stack(past_schema_items_attention_weights)

        # (group_size, action_embedding_dim)
        # _output_projection_layer is a nn (output_dim + encoder_output_dim, action_embedding_dim) which is (1200,200) when there is gnn
        # action_query conbine a lot of information not just the action.
        projected_query = self._activation(self._output_projection_layer(action_query))
        predicted_action_embeddings = self._dropout(projected_query)

        if self._add_action_bias:
            # NOTE: It's important that this happens right before the dot product with the action
            # embeddings.  Otherwise this isn't a proper bias.  We do it here instead of right next
            # to the `action_embeddings.mm` below just so we only do it once for the whole group.

            # The dimension of the action embeding () is 201 if self._add_action_bias is True, else is 200. See code in spider_parser.py 
            # So here need to add one more dim to predicted_action_embeddings to calc the dot product or add.
            # But I still do not know why the bias should add one dim. ???
            ones = predicted_action_embeddings.new([[1] for _ in range(group_size)])
            predicted_action_embeddings = torch.cat([predicted_action_embeddings, ones], dim=-1)
            
        return {
                'hidden_state': hidden_state,               # From LSTM decoder
                'memory_cell': memory_cell,                 # From LSTM decoder
                'attended_question': attended_question,     # Just original rnn_state.attended_input without any change. But Initial value is Random.
                'attention_weights': attention_weights,     # Calc from hidden_state now and original encoder_outputs 
                'past_schema_items_attention_weights': past_schema_items_attention_weights, # None in the first step
                'predicted_action_embeddings': predicted_action_embeddings,                 #
                }



    @overrides
    def _compute_action_probabilities(self,
                                      state: GrammarBasedState,
                                      hidden_state: torch.Tensor,
                                      attention_weights: torch.Tensor,
                                      past_schema_items_attention_weights: torch.Tensor,
                                      predicted_action_embeddings: torch.Tensor
                                      ) -> Dict[int, List[Tuple[int, Any, Any, Any, List[int]]]]:
        # In this section we take our predicted action embedding and compare it to the available
        # actions in our current state (which might be different for each group element).  For
        # computing action scores, we'll forget about doing batched / grouped computation, as it
        # adds too much complexity and doesn't speed things up, anyway, with the operations we're
        # doing here.  This means we don't need any action masks, as we'll only get the right
        # lengths for what we're computing.
        """
        state:
            original combined state.
        hidden_state:
            decoder hidden_state
        attention_weights:
            decoder attention_weights
        past_schema_items_attention_weights:
            decoder past_schema_items_attention_weights but initial value is none.
        predicted_action_embeddings:
            decoder hidden_state + attended_question + 1???. attended_question calc from attention_weights + encoder_outputs
        """

        group_size = len(state.batch_indices)

        # The function name in here is ambiguous. Actually, here only get the next action.
        # For the first step, it will get the action start with 'statement', 
        # including: 'statement -> [query, iue, query]' and 'statement -> [query]'.
        # So the actions[0] is: ( PS: [0] means the first element in the list obj actions)
        # {
        #    'global':(
        #                   tensor_1,
        #                   tensor_2,
        #                   list_1     # (about these three obj please see _create_grammar_state function in spider_parser.py)
        #     )
        # }
        actions = state.get_valid_actions() # all actions.

        batch_results: Dict[int, List[Tuple[int, Any, Any, Any, List[int]]]] = defaultdict(list)
        for group_index in range(group_size): # Batch
            batch_id = state.batch_indices[group_index]
            instance_actions = actions[group_index]
            predicted_action_embedding = predicted_action_embeddings[group_index]
            embedded_actions: List[int] = []

            output_action_embeddings = None
            embedded_action_logits = None
            current_log_probs = None
            linked_action_logits_encoder = None
            linked_action_ent2ent_logits = None

            if 'global' in instance_actions:
                action_embeddings, output_action_embeddings, embedded_actions = instance_actions['global']
                # This is just a matrix product between a (num_actions, embedding_dim) matrix and an
                # (embedding_dim, 1) matrix. The shape of embedded_action_logits is (num_actions).
                # embedded_action_logits is the similarity between the predicted_action_embedding and action_embeddings.
                embedded_action_logits = action_embeddings.mm(predicted_action_embedding.unsqueeze(-1)).squeeze(-1)
                action_ids = embedded_actions

            if 'linked' in instance_actions:
                linking_scores, type_embeddings, linked_actions, entity_action_linking_scores = instance_actions['linked']
                action_ids = embedded_actions + linked_actions
                # (num_question_tokens, 1)

                # for linked actions, in addition to the linking score with the attended question word, we add
                # a linking score with an attended previously decoded linked action
                # num_decoded_entities = 3
                if past_schema_items_attention_weights[group_index] is not None:
                    past_items_attention_weights, past_items_action_ids = past_schema_items_attention_weights[group_index]
                    past_schema_items_ids = [state.action_entity_mapping[batch_id][a]
                                             for a in past_items_action_ids]

                    # we are only interested about the scores of the entities the decoder has already output
                    past_entity_linking_scores = entity_action_linking_scores[:, past_schema_items_ids]

                    linked_action_ent2ent_logits = past_entity_linking_scores.mm(
                        past_items_attention_weights.unsqueeze(-1)).squeeze(-1)
                    linked_action_ent2ent_logits = self._ent2ent_ff(linked_action_ent2ent_logits.unsqueeze(-1)).squeeze(1)
                else:
                    linked_action_ent2ent_logits = 0

                linked_action_logits_encoder = linking_scores.mm(attention_weights[group_index].unsqueeze(-1)).squeeze(-1)
                linked_action_logits = linked_action_logits_encoder + linked_action_ent2ent_logits

                # The `output_action_embeddings` tensor gets used later as the input to the next
                # decoder step.  For linked actions, we don't have any action embedding, so we use
                # the entity type instead.
                if output_action_embeddings is not None:
                    output_action_embeddings = torch.cat([output_action_embeddings, type_embeddings], dim=0)
                else:
                    output_action_embeddings = type_embeddings

                if embedded_action_logits is not None:
                    action_logits = torch.cat([embedded_action_logits, linked_action_logits], dim=-1)
                else:
                    action_logits = linked_action_logits
                current_log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)
            elif not instance_actions:
                action_ids = None
                current_log_probs = float('inf')
            else: # "global" will run this else:
                # Calc the action probability in the next global action.
                action_logits = embedded_action_logits
                current_log_probs = torch.nn.functional.log_softmax(action_logits, dim=-1)

            # This is now the total score for each state after taking each action.  We're going to
            # sort by this later, so it's important that this is the total score, not just the
            # score for the current action.
            # Initial state.score is 0.
            log_probs = state.score[group_index] + current_log_probs
            batch_results[state.batch_indices[group_index]].append((group_index,
                                                                    log_probs,
                                                                    current_log_probs,
                                                                    output_action_embeddings, # As action_embedding in _construct_next_states
                                                                    action_ids,
                                                                    linked_action_logits_encoder,
                                                                    linked_action_ent2ent_logits))
        return batch_results



    def _construct_next_states(self,
                               state: GrammarBasedState,
                               updated_rnn_state: Dict[str, torch.Tensor],
                               batch_action_probs: Dict[int, List[Tuple[int, Any, Any, Any, List[int]]]],
                               max_actions: int,
                               allowed_actions: List[Set[int]]):
        """
        state:
            Initial state. encoder state.

        updated_rnn_state:
            It is created from _update_decoder_state function. updated_rnn_state is the decoder state.

        batch_action_probs:
            The probability of next action.
        
        max_actions:
            beam_size. It is 1 during training.

        allowed_actions:
            Correct results for supervision.
        """
        # pylint: disable=no-self-use

        # We'll yield a bunch of states here that all have a `group_size` of 1, so that the
        # learning algorithm can decide how many of these it wants to keep, and it can just regroup
        # them later, as that's a really easy operation.
        #
        # We first define a `make_state` method, as in the logic that follows we want to create
        # states in a couple of different branches, and we don't want to duplicate the
        # state-creation logic.  This method creates a closure using variables from the method, so
        # it doesn't make sense to pull it out of here.

        # Each group index here might get accessed multiple times, and doing the slicing operation
        # each time is more expensive than doing it once upfront.  These three lines give about a
        # 10% speedup in training time.
        group_size = len(state.batch_indices)

        chunk_index = 1 if self._num_layers > 1 else 0
        # Make tensor become list according to batch index.
        hidden_state = [x.squeeze(chunk_index)
                        for x in updated_rnn_state['hidden_state'].chunk(group_size, chunk_index)]
        memory_cell = [x.squeeze(chunk_index)
                       for x in updated_rnn_state['memory_cell'].chunk(group_size, chunk_index)]

        attended_question = [x.squeeze(0) for x in updated_rnn_state['attended_question'].chunk(group_size, 0)]

        ########################   make_state   ########################
        def make_state(group_index: int,
                       action: int,
                       new_score: torch.Tensor,
                       action_embedding: torch.Tensor) -> GrammarBasedState:
            """
            group_index:
                batch index
            action:
                next action index (created based on the allowed actions[supervision])
            new_score:
                log probability of the next action. So it must be a negative score.
            action_embedding:
                come from 'output_action_embeddings' in '_compute_action_probabilities'
            """
            batch_index = state.batch_indices[group_index]

            decoder_outputs = state.rnn_state[group_index].decoder_outputs
            is_linked_action = not state.possible_actions[batch_index][action][1]
            if is_linked_action:
                if decoder_outputs is None:
                    decoder_outputs = hidden_state[group_index].unsqueeze(0), [action]
                else:
                    decoder_outputs_states, decoder_outputs_ids = decoder_outputs
                    decoder_outputs = torch.cat((
                        decoder_outputs_states,
                        hidden_state[group_index].unsqueeze(0)
                    ), dim=0), decoder_outputs_ids + [action]

            new_rnn_state = RnnStatelet(hidden_state[group_index],
                                        memory_cell[group_index],
                                        action_embedding,
                                        attended_question[group_index],
                                        state.rnn_state[group_index].encoder_outputs,
                                        state.rnn_state[group_index].encoder_output_mask,
                                        decoder_outputs)
            for i, _, current_log_probs, _, actions, lsq, lsp in batch_action_probs[batch_index]:
                if i == group_index:
                    considered_actions = actions
                    probabilities = current_log_probs.exp().cpu()
                    considered_lsq = lsq
                    considered_lsp = lsp
                    break
            return state.new_state_from_group_index(group_index,
                                                    action,
                                                    new_score,
                                                    new_rnn_state,
                                                    considered_actions,
                                                    probabilities,
                                                    updated_rnn_state['attention_weights'],
                                                    considered_lsq,
                                                    considered_lsp
                                                    )
        ########################   make_state   ########################

        new_states = []
        for _, results in batch_action_probs.items():
            if allowed_actions and not max_actions: # max_actions = 0 or none means no beam search.
                # If we're given a set of allowed actions, and we're not just keeping the top k of
                # them, we don't need to do any sorting, so we can speed things up quite a bit.
                for group_index, log_probs, _, action_embeddings, actions in results:
                    for log_prob, action_embedding, action in zip(log_probs, action_embeddings, actions):
                        if action in allowed_actions[group_index]:
                            new_states.append(make_state(group_index, action, log_prob, action_embedding))
            else:
                # In this case, we need to sort the actions.  We'll do that on CPU, as it's easier,
                # and our action list is on the CPU, anyway.
                group_indices = []
                group_log_probs: List[torch.Tensor] = []
                group_action_embeddings = []
                group_actions = []
                for group_index, log_probs, _, action_embeddings, actions, _, _ in results:
                    if not actions:
                        continue

                    group_indices.extend([group_index] * len(actions))
                    group_log_probs.append(log_probs)
                    group_action_embeddings.append(action_embeddings)
                    group_actions.extend(actions)
                    
                if len(group_log_probs) == 0:
                    continue

                log_probs = torch.cat(group_log_probs, dim=0)
                action_embeddings = torch.cat(group_action_embeddings, dim=0)
                log_probs_cpu = log_probs.data.cpu().numpy().tolist()

                # Run the 'for' and then run 'if'. if the 'if' is True, give the tuple to the list.
                # Only keep the allowed_actions for allowed_act_in_one_case. So allowed_act_in_one_case is the correct state.
                allowed_act_in_one_case = [(log_probs_cpu[i],
                                 group_indices[i],
                                 log_probs[i],
                                 action_embeddings[i],
                                 group_actions[i])
                                for i in range(len(group_actions))
                                if (not allowed_actions or
                                    group_actions[i] in allowed_actions[group_indices[i]])]

                # We use a key here to make sure we're not trying to compare anything on the GPU.
                # Notice: Although 
                allowed_act_in_one_case.sort(key=lambda x: x[0], reverse=True) # reverse=True: From big to small.
                assert len(allowed_act_in_one_case) == 1 # This is for test. I think it is hard to have several allowed_act_in_one_case. 
                
                if max_actions: # max_actions is 1 in training, and python allow the max_actions bigger than the len(allowed_act_in_one_case)
                    allowed_act_in_one_case = allowed_act_in_one_case[:max_actions]
                for _, group_index, log_prob, action_embedding, action in allowed_act_in_one_case:
                    new_states.append(make_state(group_index, action, log_prob, action_embedding))
        return new_states

    def attend(self,
               attention: Attention,
               query: torch.Tensor,
               key: torch.Tensor,
               value: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a query (which is typically the decoder hidden state), compute an attention over the
        output of the question encoder, and return a weighted sum of the question representations
        given this attention.  We also return the attention weights themselves.

        This is a simple computation, but we have it as a separate method so that the ``forward``
        method on the main parser module can call it on the initial hidden state, to simplify the
        logic in ``take_step``.

        Be carefull, here do not support the mask.

        Normally, the attention is the dot production attention. 
        If it is DotProductAttention, the calculation steps are:
            re = key.bmm(query.unsqueeze(-1)).squeeze(-1)
            re -> softmax. Normalize the probability. And then we can get the attention_weights.
            attended_question = util.weighted_sum(value, attention_weights)
        """
        # (group_size, question_length)
        attention_weights = attention(query, key, None) # The None should be the mask. But there is not mask in here.

        if value is None:
            return attention_weights

        # (group_size, encoder_output_dim)
        attended_question = util.weighted_sum(value, attention_weights)
        return attended_question, attention_weights
