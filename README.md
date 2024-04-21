# Relational_GNN


## Aim of This Project:

Experiment with relational graph neural networks using the torch_geometric package 


## Install Instructions 

Install pytorch dependency first separately (when installing if you install torch_geometric first it may install really old versions of pytorch and cuda)

then install the remaining packages listed in requirements.txt using whatever package manager you prefer. Here is an example for pip installation 

```
pip install -r requirements. txt
```


## Running the Provided Experiments 

Each python file in the project directory relates to an experiment for that respective type of GNN. The order in which you run the files does not matter. Both ensure and will download the datasets required for each experiment. Each experiment is a classification task predicting the associated class for a node in the graph. Model weights are not stored. Accuracy is printed in the console below. Classification accuracy reported in the paper is the highest testing accuracy out of each epoch. 

To change the experiment so it is choosing a different dataset or model adjust either input arguements in the file. Which is located under the imports at the top of each file.

```
parser.add_argument('--dataset', type=str, default='AIFB',
                    choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
parser.add_argument('--model_name', type=str, default='RGCN_s',
                    choices=['RGCN', 'RGCN_s'])
```

Adjusting `--model_name` will change the model used during training and testing. Whereas adjusting `--dataset` changes the input data used for trainging and testing.
