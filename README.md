## Environment Prepare
- Please install the packages in requirements.txt.
- To calculate the spice metric: Download [Standford-coreNLP package](https://stanfordnlp.github.io/CoreNLP/), put the jar files in evaluation/spice/lib
## Dataset Prepare
### Download NMI-WSI Report dataset
Please download the NMI-WSI dataset from this link : https://www.nature.com/articles/s42256-019-0052-1#data-availability. We use the "Report" directory only. Please divide the images in test_annotation.json into valid dataset and test dataset(according to WSI names), then respectively use a file to record the image name in valid dataset and test dataset. Please arrange the files in following structure: 
- Images
    - xxx.png
    - xxx.png
    ...
- train_annotation.json
- test_annotation.json
- val.txt
- test.txt

### Generate cell graphs
run generate_cell_graphs.py

## Train model
Please tune the member variable of "GinConfig" in caption_configuration.py, then run ``python gin_caption_train.py``.

## Test model
- Set the model checkpoint path in gin_caption_test.py, then run ``python gin_caption_test.py``. This shows the text generation scores and generates a csv file includes groundtruth and generated texts of each image. 
- run ``python parse_result`` to see the lesion recognition results.

## Visulization
- Set the model checkpoint path in transformer_MM_viz.py, then run ``python transformer_MM_viz.py`` to generate visulizations described in the paper.
- Also, visulization with grad cam is also available, by doing the same steps with graph_caption_visualize.py.

## Citation
```
@article{zhou2023gnnformer,
  title={GNNFormer: A Graph-based Framework for Cytopathology Report Generation},
  author={Zhou, Yang-Fan and Yao, Kai-Lang and Li, Wu-Jun},
  journal={arXiv preprint arXiv:2303.09956},
  year={2023} }
```
