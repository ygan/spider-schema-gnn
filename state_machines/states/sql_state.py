import copy
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SqlState:
    def __init__(self,
                 possible_actions,
                 enabled: bool=True):
        """
        Will be called by new_state_from_group_index() in GrammarBasedState obj.

        One SQL correspond to one SqlState obj which include all actions (string).
        Besides, One SQL correspond to one GrammarStatelet which include all actions (tensor). 
        The attribute of valid_actions in GrammarStatelet is ``Dict[str, ActionRepresentation]`` 
        and ActionRepresentation is tensor. 
        GrammarStatelet: 1. tensor of action; 2. when to end and decoding step control based on GrammarStatelet.nonterminal_stack

        possible_actions:
            All action for the current SQL using list-type-action (see spider.py)

        enabled:
            Normally set True
        """
        self.possible_actions = [a[0] for a in possible_actions] # Include global action and specific action
        self.action_history = []
        self.tables_used = set()
        self.tables_used_by_columns = set()
        self.current_stack = []
        self.subqueries_stack = []
        self.enabled = enabled

    def take_action(self, production_rule: str) -> 'SqlState':
        """
        Update the 
        1.tables_used, 2.tables_used_by_columns, 
        3.subqueries_stack,
            store current SqlState
        4.action_history,
            store production_rule string
        5.current_stack
            store list type of production_rule
        """

        if not self.enabled:
            return self

        new_sql_state = copy.deepcopy(self)

        lhs, rhs = production_rule.split(' -> ')
        rhs_tokens = rhs.strip('[]').split(', ')
        if lhs == 'table_name':
            new_sql_state.tables_used.add(rhs_tokens[0].strip('"'))
        elif lhs == 'column_name':
            new_sql_state.tables_used_by_columns.add(rhs_tokens[0].strip('"').split('@')[0])
        elif lhs == 'iue': # Start a new SELECT, so clean these two set.
            new_sql_state.tables_used_by_columns = set()
            new_sql_state.tables_used = set()
        elif lhs == "source_subq": # Start a new SELECT, so clean these two set.ß
            new_sql_state.subqueries_stack.append(copy.deepcopy(new_sql_state))
            new_sql_state.tables_used = set()
            new_sql_state.tables_used_by_columns = set()

        new_sql_state.action_history.append(production_rule)

        new_sql_state.current_stack.append([lhs, []])

        for token in rhs_tokens:
            is_terminal = token[0] == '"' and token[-1] == '"'
            if not is_terminal:
                new_sql_state.current_stack[-1][1].append(token)

        while len(new_sql_state.current_stack[-1][1]) == 0:
            # when the rhs_tokens is terminal, it will run here.
            # For example: 'select_with_distinct -> ["select"]' enter the current_stack will become:
            # ["select_with_distinct" , []]. Because, "select" is nonterminal. So stack do not store it.
            finished_item = new_sql_state.current_stack[-1][0]
            del new_sql_state.current_stack[-1]
            if finished_item == 'statement':
                break

            if new_sql_state.current_stack[-1][1][0] == finished_item:
                # Remove [select_with_distinct , []] for exit the while loop.
                new_sql_state.current_stack[-1][1] = new_sql_state.current_stack[-1][1][1:]

            if finished_item == 'source_subq':
                new_sql_state.tables_used = new_sql_state.subqueries_stack[-1].tables_used
                new_sql_state.tables_used_by_columns = new_sql_state.subqueries_stack[-1].tables_used_by_columns
                del new_sql_state.subqueries_stack[-1]

        return new_sql_state

    def get_valid_actions(self, valid_actions: dict):
        """
        Main entrance of this function is _compute_action_probabilities in attend_past_schema_items_transition.py
        The valid_actions here is the tensor type of actions.
        For example, in the first step of decoing, it will start from 'statement', 
        so the valid_actions are the following two actions but in a tensor style:
        statement -> [query, iue, query]
        statement -> [query]

        """

        if not self.enabled:
            return valid_actions

        valid_actions_ids = []
        for key, items in valid_actions.items():
            valid_actions_ids += [(key, rule_id) for rule_id in valid_actions[key][2]]
        
        # String type of valid_actions 
        valid_actions_rules = [self.possible_actions[rule_id] for rule_type, rule_id in valid_actions_ids]

        # Init a empty actions_to_remove set. k (key) in here is only the 'global' and 'linked'
        # So there are 'global' actions_to_remove and 'linked' actions_to_remove 
        # If the right head side of the action is not in _get_current_open_clause(), we will not remove any actions. 
        actions_to_remove = {k: set() for k in valid_actions.keys()}

        # original current_clause is None.
        current_clause = self._get_current_open_clause()

        if current_clause:
            # print(type(current_clause))
            assert type(current_clause) == str

        # I think this whole complex 'if' structure is designed to prevent some weird situation and even some wrong situation.  
        if current_clause in ['where_clause', 'orderby_clause', 'join_condition', 'groupby_clause']:
            for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                rule_type, rule_id = rule_id
                lhs, rhs = rule.split(' -> ')
                rhs_values = rhs.strip('[]').split(', ')
                if lhs == 'column_name':
                    rule_table = rhs_values[0].strip('"').split('@')[0]
                    if rule_table not in self.tables_used:
                        actions_to_remove[rule_type].add(rule_id)

                # if len(self.current_stack[-1][1]) < 2:
                #     # disable condition clause when same tables
                #     rule_table = rhs_values[0].strip('"').split('@')[0]
                #     last_table = self.action_history[-1].split(' -> ')[1].strip('[]"').split('@')[0]
                #     if rule_table == last_table:
                #         actions_to_remove[rule_type].add(rule_id)

        elif current_clause in ['join_clause']:
            for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                rule_type, rule_id = rule_id
                lhs, rhs = rule.split(' -> ')
                rhs_values = rhs.strip('[]').split(', ')
                if lhs == 'table_name':
                    candidate_table = rhs_values[0].strip('"')

                    if current_clause == 'join_clause' and len(self.current_stack[-1][1]) == 2:
                        if candidate_table in self.tables_used:
                            # trying to join an already joined table
                            actions_to_remove[rule_type].add(rule_id)

                    if 'join_clauses' not in self.current_stack[-2][1] and not self.current_stack[-2][0].startswith('join_clauses'):
                        # decided not to join any more tables
                        remaining_joins = self.tables_used_by_columns - self.tables_used
                        if len(remaining_joins) > 0 and candidate_table not in self.tables_used_by_columns:
                            # trying to select a single table but used other table(s) in columns
                            actions_to_remove[rule_type].add(rule_id)

        elif current_clause in ['select_core']:
            for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                rule_type, rule_id = rule_id
                lhs, rhs = rule.split(' -> ')
                rhs_values = rhs.strip('[]').split(', ')
                if self.current_stack[-1][1][0] == 'from_clause' or self.current_stack[-1][1][0] == 'join_clauses':
                    all_tables = set([a.split(' -> ')[1].strip('[]\"') for a in self.possible_actions if
                                      a.startswith('table_name ->')])
                    if len(self.tables_used_by_columns - self.tables_used) > 1:
                        # selected columns from more tables than selected, must join
                        if 'join_clauses' not in rhs:
                            actions_to_remove[rule_type].add(rule_id)
                    if len(all_tables - self.tables_used) <= 1:
                        # don't join 2 tables because otherwise there will be no more tables to join
                        # (assuming no joining twice and no sub-queries)
                        if 'join_clauses' in rhs:
                            actions_to_remove[rule_type].add(rule_id)
                if lhs == "table_name" and self.current_stack[-1][0] == "single_source":
                    candidate_table = rhs_values[0].strip('"')
                    if len(self.tables_used_by_columns) > 0 and candidate_table not in self.tables_used_by_columns:
                        # trying to select a single table but used other table(s) in columns
                        actions_to_remove[rule_type].add(rule_id)

                if lhs == 'single_source' and len(self.tables_used_by_columns) == 0 and rhs.strip('[]') == 'source_subq':
                    # prevent cases such as "select count ( * ) from ( select city.district from city ) where city.district = ' value '"
                    search_stack_pos = -1
                    while self.current_stack[search_stack_pos][0] != 'select_core':
                        # note - should look for other "gateaways" here (i.e. maybe this is not a dead end, if there is
                        # another source_subq. This is ignored here
                        search_stack_pos -= 1
                    if self.current_stack[search_stack_pos][1][-1] == 'where_clause':
                        # planning to add where/group/order later, but no columns were ever selected
                        actions_to_remove[rule_type].add(rule_id)

                    while self.current_stack[search_stack_pos][0] != 'query':
                        search_stack_pos -= 1
                    if 'orderby_clause' in self.current_stack[search_stack_pos][1]:
                        actions_to_remove[rule_type].add(rule_id)
                    if 'groupby_clause' in self.current_stack[search_stack_pos][1]:
                        actions_to_remove[rule_type].add(rule_id)
        else: # current_clause is None 
            pass

        new_valid_actions = {}
        new_global_actions = self._remove_actions(valid_actions, 'global',
                                                  actions_to_remove['global']) if 'global' in valid_actions else None
        new_linked_actions = self._remove_actions(valid_actions, 'linked',
                                                  actions_to_remove['linked']) if 'linked' in valid_actions else None

        if new_linked_actions is not None:
            new_valid_actions['linked'] = new_linked_actions
        if new_global_actions is not None:
            new_valid_actions['global'] = new_global_actions

        # if len(new_valid_actions) == 0 and valid_actions:
        #     # should not get here! implies that a rule should have been disabled in past (bug in this parser)
        #     # log and do not remove rules (otherwise crashes)
        #     # logger.warning("No valid action remains, error in sql decoding parser!")
        #     # logger.warning("Action history: " + str(self.action_history))
        #     # logger.warning("Tables in db: " + ', '.join([a.split(' -> ')[1].strip('[]\"') for a in self.possible_actions if a.startswith('table_name ->')]))
        #
        #     return valid_actions

        # It will be the same as the input valid_actions in first round because there no need to remove any actions.
        return new_valid_actions

    @staticmethod
    def _remove_actions(valid_actions, key, ids_to_remove):
        if len(ids_to_remove) == 0:
            return valid_actions[key]

        if len(ids_to_remove) == len(valid_actions[key][2]):
            return None

        current_ids = valid_actions[key][2]
        keep_ids = []
        keep_ids_loc = []

        for loc, rule_id in enumerate(current_ids):
            if rule_id not in ids_to_remove:
                keep_ids.append(rule_id)
                keep_ids_loc.append(loc)

        items = list(valid_actions[key])
        items[0] = items[0][keep_ids_loc]
        items[1] = items[1][keep_ids_loc]
        items[2] = keep_ids

        if len(items) >= 4:
            items[3] = items[3][keep_ids_loc]
        return tuple(items)

    def _get_current_open_clause(self):
        # Actually this function is get the closest relevant_clauses from self.current_stack.

        relevant_clauses = [
            'where_clause',  # where_clause -> ["where", expr, where_conj]
                             # where_clause -> ["where", expr]
            'orderby_clause',
            'join_clause',
            'join_condition',
            'select_core',
            'groupby_clause',# groupby_clause -> ["group", "by", group_clause, "having", expr]
                             # groupby_clause -> ["group", "by", group_clause]
            'source_subq'    # source_subq -> ["(", query, ")"] # I can not remove 'source_subq'. 
                             # although in get_valid_actions function, we do not use this token. 
                             # But it can use it to stop the following search code to avoid some problems.
                             # 'source_subq' is the start of a subsql, so it may cause conflict with 'where_clause' if we remove it here. 
        ]

        # Search for the key words appearing in relevant_clauses from end to start.
        # Once find the relevant_clauses, return it. So it return the most closest relevant_clauses from current_stack.
        for rule in self.current_stack[::-1]: 
            if rule[0] in relevant_clauses:
                return rule[0]

        return None
