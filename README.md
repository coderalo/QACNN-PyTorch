# QACNN-PyTorch
This is the PyTorch implementation of QACNN (https://arxiv.org/abs/1709.05036).  
Basically, you can train this model on arbitrary QA dataset; since the original purpose of this implementation is to do experiments on Chinese dataset, you may find some code snippets not suitable for English dataset, please feel free to modify them.  
*The default training configuration in this repo is not similar with experiments in paper; if you want to reproduce the result, you might have to change the configuration in ``config.yaml``.*
  
The file num2chinese.py is modified from [here](https://gist.github.com/gumblex/0d65cad2ba607fd14de7).
## Usage
You can change the training / testing configuration by editing ``config.yaml``.  
First, you have to make sure that under your ``data_dir`` existing three folders: ``train``, ``valid``, ``test``. The data files should be put under these three folders in csv format like:
```
id,passage,question,choice1,choice2,choice3,choice4,answer
0,xxx,ooo,aaa,bbb,ccc,ddd,1
...
```
You can split the data files into several csv files, but you have to make sure that the header exists, and the order of these fields are same as above. For files under ``test/``, the ``answer`` field is not necessary.
  
For training, just run the file ``train.py``; for testing, you can specify the checkpoint on your own, or just use the best checkpoint (that is, ``best.ckpt``) under the experiment directory (in ``config.yaml``).  
If you want to use pretrained embeddings, you have to specify the argument ``embedding_dir``, and put the repsective files under it with name like ``100.vec`` so that it can be found automatically.  
The format of embedding files should be like below:
```
the 0.12 0.11 ...
he 0.33 0.98 ...
...
```
