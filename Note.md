# Study Note For Repository
Tutorial:

https://allennlp.org/tutorials
https://mlexplained.com/2019/01/30/an-in-depth-tutorial-to-allennlp-from-basics-to-elmo-and-bert/


## Structure:

AllenNLP base on:
1. Load/Read data.
2. Train.
3. Test/Verify.

## Load/Read Data:

### Define Data Reader

The data reader read data through the following configuration:
```
"dataset_reader": {
    "type": "spider",
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "lazy": false,
    "keep_if_unparsable": false,
    "loading_limit": -1
  },

  "validation_dataset_reader": {
    "type": "spider",
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "lazy": false,
    "keep_if_unparsable": true,
    "loading_limit": -1
  },
```
So we know that there will be two data reader that are the same type (spider).

The code:  `@DatasetReader.register("spider")`  tell us that `class SpiderDatasetReader(DatasetReader)` is the data reader we need. And we will create two `SpiderDatasetReader` objective.

Other configuration parameter except `type` will be sent to the constructor of `SpiderDatasetReader`, such as `tables_file` and `dataset_path`.

### Read Data
After construct the data reader, AllenNLP will call def `_read(self, file_path: str)` to read the data automatically. And then we can finish the process of reading data.


