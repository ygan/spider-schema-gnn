import re
from collections import Set, defaultdict
from typing import Dict, Tuple, List

from allennlp.data import Tokenizer, Token
from ordered_set import OrderedSet
from unidecode import unidecode

from dataset_readers.dataset_util.spider_utils import TableColumn, read_dataset_schema, read_dataset_values
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph


# == stop words that will be omitted by ContextGenerator
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


class SpiderDBContext:
    schemas = {}
    db_knowledge_graph = {}
    db_tables_data = {}

    def __init__(self, db_id: str, utterance: str, tokenizer: Tokenizer, tables_file: str, dataset_path: str):
        self.dataset_path = dataset_path
        self.tables_file = tables_file
        self.db_id = db_id
        self.utterance = utterance

        # lemma is the basic form of a word, 
        # for example the singular form of a noun or the infinitive form of a verb, 
        # as it is shown at the beginning of a dictionary entry 
        tokenized_utterance = tokenizer.tokenize(utterance.lower())

        # For example: if the utterance.lower() = ['biggest', 'departments']
        # tokenized_utterance will be [token_from_('biggest'), token_from_('departments')]
        # And token_from_('biggest').text = 'biggest', token_from_('biggest').lemma_ = 'big';
        # And token_from_('departments').text = 'departments', token_from_('departments').lemma_ = 'department';
        
        # the obj Token is similar to the obj in tokenized_utterance but not the same.
        # And the here, we take only a part of data from original tokenized_utterance.
        # So the Token obj is a simplified version of the obj in tokenized_utterance
        self.tokenized_utterance = [Token(text=t.text, lemma=t.lemma_) for t in tokenized_utterance]

        if db_id not in SpiderDBContext.schemas:
            SpiderDBContext.schemas = read_dataset_schema(self.tables_file)
        self.schema = SpiderDBContext.schemas[db_id]

        self.knowledge_graph = self.get_db_knowledge_graph(db_id)

        entity_texts = [self.knowledge_graph.entity_text[entity].lower()
                        for entity in self.knowledge_graph.entities]
        entity_tokens = tokenizer.batch_tokenize(entity_texts)
        self.entity_tokens = [[Token(text=t.text, lemma=t.lemma_) for t in et] for et in entity_tokens]

    @staticmethod
    def entity_key_for_column(table_name: str, column: TableColumn) -> str:
        if column.foreign_key is not None:
            column_type = "foreign"
        elif column.is_primary_key:
            column_type = "primary"
        else:
            column_type = column.column_type
        return f"column:{column_type.lower()}:{table_name.lower()}:{column.name.lower()}"




    def get_db_knowledge_graph(self, db_id: str) -> KnowledgeGraph:
        entities: Set[str] = set()
        neighbors: Dict[str, OrderedSet[str]] = defaultdict(OrderedSet)
        entity_text: Dict[str, str] = {}
        foreign_keys_to_column: Dict[str, str] = {}

        db_schema = self.schema
        tables = db_schema.values()

        if db_id not in self.db_tables_data:
            self.db_tables_data[db_id] = read_dataset_values(db_id, self.dataset_path, tables)

        tables_data = self.db_tables_data[db_id]

        string_column_mapping: Dict[str, set] = defaultdict(set)

        for table, table_data in tables_data.items():
            for table_row in table_data:
                for column, cell_value in zip(db_schema[table.name].columns, table_row):
                    if column.column_type == 'text' and type(cell_value) is str:
                        cell_value_normalized = self.normalize_string(cell_value)
                        column_key = self.entity_key_for_column(table.name, column)
                        string_column_mapping[cell_value_normalized].add(column_key)

        # string_entities because it only search the text column data.
        # string_entities is the column information that its value appearing in the question.
        string_entities = self.get_entities_from_question(string_column_mapping)

        # table.text|column.text -> table or column name
        # table_key  -> 'table':table.text
        # entity_key -> 'column':column_type:table.text:column.text. column_type can be: 'text', 'number', 'primary', 'foreign'!!!
        # entities   ->  set of table_key and entity_key. The number of items is number of table + number of columns in each table. Table name + column name.
        # entity_text->  dict of {table_key:table.text} and {entity_key:column.text}
        # neighbors  ->  dict of {entity_key:table.text} and {table_key:{all its column.text}}
        for table in tables:
            table_key = f"table:{table.name.lower()}"
            entities.add(table_key)
            entity_text[table_key] = table.text
            for column in db_schema[table.name].columns:
                entity_key = self.entity_key_for_column(table.name, column)
                entities.add(entity_key)
                neighbors[entity_key].add(table_key)
                neighbors[table_key].add(entity_key)
                entity_text[entity_key] = column.text

        # token_in_utterance-> the token in utterance that appear in the data of database
        # string_entity     -> type:token_in_utterance. type is only 'string' here.
        # column_keys       -> list of entity_key
        # column_key        -> entity_key
        # Now, entities     -> set of table_key and entity_key and string_entity
        # Now, entity_text  -> dict of {table_key:table_names} and {entity_key:column_names}. table_names and column_names come from tables.json. Not table_names_original and column_names_original!!
        # Now, neighbors    -> Plus: {string_entity:all its entity_key} and {entity_key:table.text and string_entity (if it has)} and {table_key:{all its column.text}}
        for string_entity, column_keys in string_entities:
            entities.add(string_entity)
            for column_key in column_keys:
                neighbors[string_entity].add(column_key)
                neighbors[column_key].add(string_entity)
            entity_text[string_entity] = string_entity.replace("string:", "").replace("_", " ")

        # loop again after we have gone through all columns to link foreign keys columns
        for table_name in db_schema.keys():
            for column in db_schema[table_name].columns:
                if column.foreign_key is None:
                    continue

                other_column_table, other_column_name = column.foreign_key.split(':')

                # must have exactly one by design
                other_column = [col for col in db_schema[other_column_table].columns if col.name == other_column_name][0]

                entity_key = self.entity_key_for_column(table_name, column)
                other_entity_key = self.entity_key_for_column(other_column_table, other_column)

                neighbors[entity_key].add(other_entity_key)
                neighbors[other_entity_key].add(entity_key)

                foreign_keys_to_column[entity_key] = other_entity_key

        # if we can not find tokens appearing in both the data of database and token of question:
        # But even we can find tokens, here still the same.
        #   1. the length of entities and neighbors and entity_text are all equal to number of table + number of columns in each table
        #   2. entities is table and column 'name'
        #   3. neighbors is the relationship for every column and table. For example:
        #           neighbors[table].items = [all its column name]
        #           neighbors[column].items = [table_name, relation_column]; 
        #               if column_1 is a primary key and column_2 and column_3 is foreign keys reference to column_1, neighbors[column_1].items = [column_1's table_name, column_2, column_3]
        #           neighbors[column_2].items = [column_2's table_name, column_1]
        #           neighbors[column_3].items = [column_3's table_name, column_1]
        #           if a column_4 does not have any relation_column, neighbors[column_4].items = [column_4's table_name];
        #   4. entity_text: dict of {table_key:table_names} and {entity_key:column_names}. table_names and column_names is a readable name different from the original name in database.
        
        # if we can find token appearing in both the data of database and token of question:
        #   1. the length of three objectives are still equal and its value = previous value + number of token found
        #       So every thing is the same as we can not find the token. The token expand the three objectives and nothing more, Now let's discuss the new data brought by the token.
        #   2. entities += ['string':token]. We call ['string':token] as type_with_token.
        #   3. neighbors += {type_with_token:[its columns]}. Supposing the token appearing in question and name_column and previous_name_column. here will be:
        #           neighbors += {type_with_token:[name_column,previous_name_column]}.
        #           neighbors[name_column].add(type_with_token), which is from neighbors[name_column] = {name_column's table_name} to neighbors[name_column] = {name_column's table_name, type_with_token}
        #           neighbors[previous_name_column] do the same process.
        #           neighbors[name_column] and neighbors[previous_name_column] exist for neighbors even we do not find tokens. But neighbors[type_with_token] only exist when we find tokens.
        #   4. entity_text +=  {type_with_token: token}
        
        # To now, we know how to create a KnowledgeGraph:
        # entities is node name in a graph.
        # entity_text is real text for the node. For example, we can call the node as 'column:text:management:temporary_acting', but its text for this node is 'temporary acting'. 
        #           'temporary acting' copy from column_names in tables.json (not column_names_original). So we will analyse the text only. Node name is just a symbol.
        # neighbors is the edge in a graph.
        #           neighbors['node_name'].items=[list of (other) node name ]. Here is absolutely other!
        #           The edge contain: table-column, foreign-primary, column-token_in_quesion.
        # Their length is number of node in this graph.
        kg = KnowledgeGraph(entities, dict(neighbors), entity_text) # node name, edge, node value

        # Add a attribute for kg.
        # Example data of foreign_keys_to_column generated from table 'department_management'
        # foreign_keys_to_column['column:foreign:management:department_id'] = 'column:primary:department:department_id'
        # foreign_keys_to_column['column:foreign:management:head_id'] = 'column:primary:head:head_id'
        kg.foreign_keys_to_column = foreign_keys_to_column

        return kg

    def _string_in_table(self, candidate: str,
                         string_column_mapping: Dict[str, set]) -> List[str]:
        """
        Checks if the string occurs in the table, and if it does, returns the names of the columns
        under which it occurs. If it does not, returns an empty list.
        """
        candidate_column_names: List[str] = []
        # First check if the entire candidate occurs as a cell.
        if candidate in string_column_mapping:
            candidate_column_names = string_column_mapping[candidate]
        # If not, check if it is a substring pf any cell value.
        if not candidate_column_names:
            for cell_value, column_names in string_column_mapping.items():
                if candidate in cell_value:
                    candidate_column_names.extend(column_names)
        candidate_column_names = list(set(candidate_column_names))
        return candidate_column_names

    def get_entities_from_question(self,
                                   string_column_mapping: Dict[str, set]) -> List[Tuple[str, str]]:
        """
        An entity is any object in the system that we want to model and store information about.
        But here is get the column name from question. For example, if the question is: "get all information of jack"
        Supposing the table in database is:{id,name,salary} and we can find 'jack' from the 'name' column.
        So we can get the column 'name' from the question because there is 'jack' in both the question and table data.
        It will return the column list.
        """

        entity_data = []
        for i, token in enumerate(self.tokenized_utterance):
            token_text = token.text
            if token_text in STOP_WORDS:
                continue
            normalized_token_text = self.normalize_string(token_text)
            if not normalized_token_text:
                continue
            token_columns = self._string_in_table(normalized_token_text, string_column_mapping)
            if token_columns:
                token_type = token_columns[0].split(":")[1]
                entity_data.append({'value': normalized_token_text,
                                    'token_start': i,
                                    'token_end': i+1,
                                    'token_type': token_type,
                                    'token_in_columns': token_columns})

        # extracted_numbers = self._get_numbers_from_tokens(self.question_tokens)
        # filter out number entities to avoid repetition
        expanded_entities = []
        for entity in self._expand_entities(self.tokenized_utterance, entity_data, string_column_mapping):
            if entity["token_type"] == "text":
                expanded_entities.append((f"string:{entity['value']}", entity['token_in_columns']))
        # return expanded_entities, extracted_numbers  #TODO(shikhar) Handle conjunctions

        return expanded_entities

    @staticmethod
    def normalize_string(string: str) -> str:
        """
        These are the transformation rules used to normalize cell in column names in Sempre.  See
        ``edu.stanford.nlp.sempre.tables.StringNormalizationUtils.characterNormalize`` and
        ``edu.stanford.nlp.sempre.tables.TableTypeSystem.canonicalizeName``.  We reproduce those
        rules here to normalize and canonicalize cells and columns in the same way so that we can
        match them against constants in logical forms appropriately.
        """
        # Normalization rules from Sempre
        # \u201A -> ,
        string = re.sub("‚", ",", string)
        string = re.sub("„", ",,", string)
        string = re.sub("[·・]", ".", string)
        string = re.sub("…", "...", string)
        string = re.sub("ˆ", "^", string)
        string = re.sub("˜", "~", string)
        string = re.sub("‹", "<", string)
        string = re.sub("›", ">", string)
        string = re.sub("[‘’´`]", "'", string)
        string = re.sub("[“”«»]", "\"", string)
        string = re.sub("[•†‡²³]", "", string)
        string = re.sub("[‐‑–—−]", "-", string)
        # Oddly, some unicode characters get converted to _ instead of being stripped.  Not really
        # sure how sempre decides what to do with these...  TODO(mattg): can we just get rid of the
        # need for this function somehow?  It's causing a whole lot of headaches.
        string = re.sub("[ðø′″€⁄ªΣ]", "_", string)
        # This is such a mess.  There isn't just a block of unicode that we can strip out, because
        # sometimes sempre just strips diacritics...  We'll try stripping out a few separate
        # blocks, skipping the ones that sempre skips...
        string = re.sub("[\\u0180-\\u0210]", "", string).strip()
        string = re.sub("[\\u0220-\\uFFFF]", "", string).strip()
        string = string.replace("\\n", "_")
        string = re.sub("\\s+", " ", string)
        # Canonicalization rules from Sempre.
        string = re.sub("[^\\w]", "_", string) # replace all non word(a-z) with "_"
        string = re.sub("_+", "_", string)
        string = re.sub("_$", "", string)
        return unidecode(string.lower())

    def _expand_entities(self, question, entity_data, string_column_mapping: Dict[str, set]):
        new_entities = []
        for entity in entity_data:
            # to ensure the same strings are not used over and over
            if new_entities and entity['token_end'] <= new_entities[-1]['token_end']:
                continue
            current_start = entity['token_start']
            current_end = entity['token_end']
            current_token = entity['value']
            current_token_type = entity['token_type']
            current_token_columns = entity['token_in_columns']

            while current_end < len(question):
                next_token = question[current_end].text
                next_token_normalized = self.normalize_string(next_token)
                if next_token_normalized == "":
                    current_end += 1
                    continue
                candidate = "%s_%s" %(current_token, next_token_normalized)
                candidate_columns = self._string_in_table(candidate, string_column_mapping)
                candidate_columns = list(set(candidate_columns).intersection(current_token_columns))
                if not candidate_columns:
                    break
                candidate_type = candidate_columns[0].split(":")[1]
                if candidate_type != current_token_type:
                    break
                current_end += 1
                current_token = candidate
                current_token_columns = candidate_columns

            new_entities.append({'token_start': current_start,
                                 'token_end': current_end,
                                 'value': current_token,
                                 'token_type': current_token_type,
                                 'token_in_columns': current_token_columns})
        return new_entities
