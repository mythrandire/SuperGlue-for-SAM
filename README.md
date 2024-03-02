# SuperGlue for SAM
Adapted from [SuperGlue-for-Visual-Place-Recognition](https://github.com/jomariya23156/SuperGlue-for-Visual-Place-Recognition)

## Dependencies
* Python 3
* PyTorch 
* OpenCV 
* Matplotlib
* NumPy 
* Pandas


## Added Contents
* `superglue_detect_cereal.py`: To be run once SAM has been used to detect cereal boxes and instance images have been found and created.

This is to be run on the instance images (which are the query images), against the input reference images (the provided zip of product thumbnails).

Input arguments:

* `--input_dir`: Path to database image directory
* `--query_dir`: Path to query image directory
* `--output_dir`: Path to store npz and visualization files

## BibTeX Citation
If you use any ideas from the paper or code from this repo, please consider citing:

```txt
@inproceedings{sarlin20superglue,
  author    = {Paul-Edouard Sarlin and
               Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  title     = {{SuperGlue}: Learning Feature Matching with Graph Neural Networks},
  booktitle = {CVPR},
  year      = {2020},
  url       = {https://arxiv.org/abs/1911.11763}
}
```

## Legal Disclaimer
Magic Leap is proud to provide its latest samples, toolkits, and research projects on Github to foster development and gather feedback from the spatial computing community. Use of the resources within this repo is subject to (a) the license(s) included herein, or (b) if no license is included, Magic Leap's [Developer Agreement](https://id.magicleap.com/terms/developer), which is available on our [Developer Portal](https://developer.magicleap.com/).
If you need more, just ask on the [forums](https://forum.magicleap.com/hc/en-us/community/topics)!
We're thrilled to be part of a well-meaning, friendly and welcoming community of millions.
