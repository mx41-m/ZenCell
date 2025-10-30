# ZenCell
ZenCell is a deep learning pipeline for large-scale 3D fluorescence microscopy. It delivers efficient and accurate cell segmentation by leveraging self-supervised pretraining, reducing dependence on manual annotations. 

## Pipeline Overview 
ZenCell consists of three main components:
- Self-supervised pretraining on large-scale unlabeled raw data
- Interactive annotation tool for efficiently collecting labels on new datasets
- Fine-tuning to adapt pretrained models with new labels for accurate 3D cell segmentation on new datasets

## Environment Installation 
Pretraining, fine-tuning, and the annotation tool share the same environment. We provide a Conda-based installation guide for setting it up in [ANNOTOOL.md](doc/ANNOTOOL.md).

## Self-supervised pretraining
This step can be skipped if you use the provided pretrained weights. However, if your data distribution differs significantly from our pretraining data, we recommend running pretraining on your own unlabeled raw images for optimal performance. The pre-training instruction is in [PRETRAIN.md](doc/PRETRAIN.md)

## Interactive annotation tool
The interactive annotation tool is built on top of the Segment Anything Model (SAM). Users can crop regions within a sample where the model underperforms and simply click on the centers of missed cells to generate new masks. This tool enables efficient collection of annotations, supporting fine-tuning on new datasets and continuous improvement of model performance. The annotation tool instruction is in [ANNOTOOL.md](doc/ANNOTOOL.md)

## Fine-tuning 
This step fine-tunes the pretrained model for 3D cell segmentation on the user’s new samples. The fine-tuning instruction is in [FINETUNE.md](doc/FINETUNE.md)

## Whole brain segmentation 
Once the fine-tuned model is ready, users can apply it to generate whole-brain 3D segmentations. The instruction in in [INFERENCE.md](doc/INFERENCE.md)

## TODO
- Provide a shortcut for model installation, training, and inference.
- Release our 3D annotation tool.
- Provide a demo for model installation and human-in-the-loop training and inference. This demo will include:
  - Installing the model and using either the pre-trained or fine-tuned checkpoint to generate 3D cell segmentations for new samples.
  - Leveraging our 3D annotation tool to efficiently add new labels.
  - Training the model with newly added samples, enabling it to further improve performance on unseen data.

