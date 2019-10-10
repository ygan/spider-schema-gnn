import json
import sys

from allennlp.commands import main


# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "predict",
    "experiments/name_of_experiment",
    # "/home/yj/python/Github/bilayerSQL/data/spider/dev.json",
    "/home/yj/backup/python/codalab/data/dev.json",
    "--predictor","spider" ,
    "--use-dataset-reader" ,
    "--cuda-device=0" ,
    "--output-file","experiments/name_of_experiment/prediction.sql" ,
    "--silent" ,
    "--include-package","models.semantic_parsing" ,#.bert_spider_parser
    "--include-package","dataset_readers.spider" ,
    "--include-package","predictors.spider_predictor" ,
    "--weights-file","experiments/name_of_experiment/best.th"
]

# from allennlp.common.params import Params
# a = Params.from_file("train_configs/defaults-all-col.jsonnet")
# from models.semantic_parsing import SpiderParser
# from allennlp.data import Vocabulary
# k = a.params["model"]
# # if "type" in k.keys():
# #     k.pop("type")
# vocab_dir = 'experiments/name_of_experiment/vocabulary'
# vocab_params = Params({})
# vocab_choice = vocab_params.pop_choice("type", Vocabulary.list_available(), True)
# vocab = Vocabulary.by_name(vocab_choice).from_files(vocab_dir)
# # k["vocab"]=vocab
# model = SpiderParser.from_params(vocab=vocab, params=a)
# sp = SpiderParser(**k)

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('--toy', action='store_true', default=False,
#         help='If set, use small data; used for fast debugging.')

main()