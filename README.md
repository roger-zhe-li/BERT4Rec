# BERT4Rec

This is my implementation and experimental data for the paper:

Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM '19). Association for Computing Machinery, New York, NY, USA, 1441–1450.

Code Author: Roger Zhe Li

## Environment Settings
I use PyTorch 2.2.1 as the main deep learning framework for implementation, with the CUDA version of 12.1. <br/>
The debugging stage relies much on the Torchsnooper package. <br/>


## File and Folder Structure

[./data_proc.py](https://github.com/roger-zhe-li/BERT4Rec/blob/main/data_proc.py): transform the tfrecord data into a Pandas dataframe. In accordance with the original paper, I filtered out users with fewer than 5 interacted items; <br/>
[./data_loader.py](https://github.com/roger-zhe-li/BERT4Rec/blob/main/data_loader.py)](https://github.com/roger-zhe-li/BERT4Rec/blob/main/data_loader.py): load and transform the raw data into PyTorch dataloaders. One loader for training, one loader for validation, and one loader for testing; <br/>
[./models.py](https://github.com/roger-zhe-li/BERT4Rec/blob/main/models.py): implement the modules consisting of the BERT4Rec model, including multi-head attention, point-wise feedforward network, sublayer connection (residual blocks for both attention and PFFN), embedding generator (both token and the LEARNABLE position embeddings), Transformer and its stack, i.e., BERT architecture. Each architecture forms one class, and the core class is *BERT*.   <br/>
[./main.py](https://github.com/roger-zhe-li/BERT4Rec/blob/main/main.py): the main executive python file. <br/>

[./recommendation](https://github.com/roger-zhe-li/BERT4Rec/tree/main/recommendation): Store the original dataset. Here I use the ./dataset/data-3.tfrecord to finish the demo. A full-scale model training process can be done by concatenating all 4 parts of the data; <br/>
[./recommendation/movie_title_by_index.json](https://github.com/roger-zhe-li/BERT4Rec/blob/main/recommendation/movie_title_by_index.json): Store the mapping between movie IDs and the real movie titles. Here I use it for getting the number of items in the dataset; <br/>


## Illustration on some key design choices
- Following the original paper, I use 2 heads and 2 stacks of the transformer encoder structure. Sequence length is set at 50 for demo. The mask probability &rou is set at 0.2, also adhering to the statement in the original paper;
- Since the item ID starts from 0 and ends at n_item - 1 (40857), here I set the mask token ID as n_item, and the padding token ID as n_item + 1. All the sequence mask and masked_fill operations all follow this ID setting;
- Evaluation is done using nDCG@10 and RR@10. Both are position-aware to avoid the disadvantage of HR@10. 10 is a magic number, which could also be further included in a list like \[3, 5, 10, 20\] for more extensive analysis on different cutoffs;
- Different from the original paper, negative sampling on validation/test samples follows a random sampler rather than a popularity-based one. This is because I think it is even more important to get more samples exposed in evaluation rather than always only put the most popular items as negative samples, which might cause potential bias issues (though yet to confirm with experimental results);
- To assure the consistency of valid/test results, the 100 negative samples for each user are fixed and only loaded once. However, to get different masked sequences of each training sample, the train loader is reinitialized each epoch. Therefore, the random seed is only fixed in PyTorch, but for numpy it is not, so that each epoch the model will see different masked sequences.


## Example to run the code
For now, the parse_arg function is still not implemented. All key hyperparameters are hard-coded. This is sub-optimal, but due to the time constraints the priority goes to first getting the model running and trained. Therefore, so far the code can be run as

```
python3 main.py
```
The code is run end-to-end successfully in both the CPU and GPU environments.

## License
* This repo is under an [MIT license]([https://creativecommons.org/licenses/by/4.0/](https://opensource.org/license/mit)).
* Software is distributed under the terms of the [MIT License](https://opensource.org/licenses/MIT).



Last Update Date: July 6, 2021
