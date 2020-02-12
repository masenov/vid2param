# vid2param

Accompanying code for [Vid2Param: Modelling of Dynamics Parameters from Video](https://masenov.com/publication/vid2param/)

[Dataset and trained weights](https://uoe-my.sharepoint.com/:f:/g/personal/s1247380_ed_ac_uk/EuE_ucgyZatJiiWG851dTcMBICOw7fggJy3jjjcwgorVQg?e=tnH1sU)

### To test the model:

Download the dataset and trained weights and paste in the root directory and run:
```
python3 main.py --train=0 --exp_name=test
```
And run tensorboard to visualize:
```
tensorboard --logdir runs --port=3000
```
Now you can explore the results in your browser at:
[http://localhost:3000/](http://localhost:3000/)

### To train the model:
```
python3 main.py --alpha=10 --exp_name=test --gpu=0
```

### Required packages:
Tested with libraries located in requirements.txt. To install:
```
pip install -r requirements.txt
```
