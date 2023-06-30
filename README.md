# KeyholeSeg

KeyholeSeg is a deep-learning-based semantic segmentation tool that is capable of automatic segmentation of keyholes in LPBF X-ray images with accurate boundaries. This project is built based on BASNet, [*Boundary-Aware Segmentation Network for Mobile and Web Application*](https://github.com/xuebinqin/BASNet), (Qin et al.). This tool is consisted of a filter, and a trained network.

## Instructions

1. Setup the workspace following [BASNet repo]((https://github.com/xuebinqin/BASNet)).
2. Download [Filter.mlx](https://github.com/WilliamDongSH/KeyholeSeg/blob/master/matlab/Filter.mlx) file, and change directories to the location of raw X-ray images. Tune parameters and run the code to acquire the filtered images. Place the filtered images into designated directory in the BASNet workspace.
3. Create a new directory /saved_model/basnet_bsi, download trained model from [google drive](https://drive.google.com/drive/u/0/folders/1rAjodTchu0Gd_FJ0Bb_QwPuHwyQY-9um), and place under the new directory.
4. Change the directory in the script [basnet_test.py](https://github.com/WilliamDongSH/KeyholeSeg/blob/master/basnet_test.py) to your location of filtered images and expected location to store segmentations. Run the script.
5. If you want to check the performaneof the segmentaion, change the directories in [checker.mlx](https://github.com/WilliamDongSH/KeyholeSeg/blob/master/matlab/checker.mlx), and run the code. It will report the IOU and BFScore of segmentations.
6. If there are significant changes in conditions, and you want to retrain the model, label the segmentation of your corresponding training inputs. Placed the training inputs and outputs in your designated directory and change the [basnet_train.py](https://github.com/WilliamDongSH/KeyholeSeg/blob/master/basnet_train.py) correspondingly. It will log the loss value of training and testing after every epoches in [basnet_train_log.csv](https://github.com/WilliamDongSH/KeyholeSeg/blob/master/basnet_train_log.csv) (now we have two separated testing sets).

