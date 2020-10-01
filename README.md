# Document-level Event Role Filler Extraction (ACL 2020)
[paper link](https://www.aclweb.org/anthology/2020.acl-main.714.pdf)

Please also check our [sibling project on event entity extraction for template filling](https://github.com/xinyadu/doc_event_entity)

## Dependencies

* python 3.5.6
* spacy==2.0.12
* torch 0.4.1 (for ```./model```)
* pytorch-pretrained-bert==0.6.2 (for ```./model```)


## Dataset

    ./data/
	├── process_train_dev.py  # proc script
	├── process_test.py       # proc script
	│ 
	├── processed/            # processed data files
	│   ├── train.json 
	│   ├── dev.json       
	│   └── test.json           
	│ 
	└── raw_muc/              # Raw data files from MUC-{3,4}

Run preprocessing for train and dev, use flag `-full` to include all the templates.
> python process\_train\_dev.py 

Run preprocessing for test,
> python process\_test.py

## Evaluation

To run the eval script:

> python eval.py --goldfile \<gold file path> --predfile \<pred file path>

We use `./data/processed/test.json` for `<gold file path>` in the experiments. We also include an example output file (`./model/pred.json`) in the model foler:

> python eval.py --goldfile ./data/processed/test.json --predfile ./model/pred.json

If you use our eval script, please make sure the `<pred file>` is of the same format as `pred.json`.

## Model Code

We also include a sample output file in the folder.


## Citation
If you use materials in this repo helpful, please cite:

```
@inproceedings{du2020doucment,
    title={Document-Level Event Role Filler Extraction Using Multi-Granularity Contextualized Encoding of the Text},
    author={Du, Xinya and Cardie, Claire},
    booktitle={Association for Computational Linguistics (ACL)},
    year={2020}
  }
```
