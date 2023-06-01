# Part of Speech Analysis Based on Hidden Markov Model
This program is a demo of implementing **Part of Speech Analysis
by Hidden Markov Model**. The database is preprocessed Remin Daily corpus which
save as JSON file.

## Run
Use command
```shell
python run.py --train
```
to train the model, which will save matrix checkpoint files into */params*.

Use command
```shell
python run.py --test --sentence="输入要处理的句子"
```
to test the model, which will show the result of analysis in terminal.

Use command
```shell
python run.py --eval
```
to generate the precision score, recall score and f1 score
based on the test data */data/pos_test.json*.