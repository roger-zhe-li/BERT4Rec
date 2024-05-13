# BERT4Rec

# New Insights into Metric Optimization for Ranking-based Recommendation

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


## Example to run the code
For now the parse_arg function is still not implemented. All key hyperparameters are hard-coded. This is sub-optimal, but due to the time constraints the priority goes to first getting the model running and trained. Therefore, so far the code can be run as

```
python3 main.py
```

```
### Dataset
The dataset is a session-based movie dataset with 4 different sub-files stored in the tfrecord format. In this repo I first turned the data into a Pandas dataframe and load it to the PyTorch dataloader.
For the sake of time, in the demo I only use the 4th part (./data-3), the smallest one, for experiments.


## License
* This repo is under an [MIT license]([https://creativecommons.org/licenses/by/4.0/](https://opensource.org/license/mit)).
* Software is distributed under the terms of the [MIT License](https://opensource.org/licenses/MIT).



Last Update Date: July 6, 2021
