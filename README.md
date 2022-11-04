paper link: http://star.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-3180/paper-182.pdf

![image](https://user-images.githubusercontent.com/49955700/199968902-fd7f61f6-ee0a-43cd-a5f2-9295a11748a5.png)


This work is sponsored by Natural Science Foundation of China(62276242), CAAI-Huawei MindSpore Open Fund(CAAIXSJLJJ-2021-016B), Anhui Province Key Research and Development Program(202104a05020007), and USTC Research Funds of the Double First-Class Initiative(YD2350002001)”。

Later, Mindspore will be provided for implementation.




## How to run

1. Obtain all Cifar10 data from the official website to the folder ==CIFAR-10-FANs-PY==

2. Create two empty folders.  ==new_data==and ==train_model== 
3. Run ==gen_corr_data.ipynb== to generate corruption data

4. Run ==final_gen_data_iter. Ipynb== to cell (# # # -- -- -- -- -- - # # # #) to stop, to generate the first generation of the data.
5. ==num = 1== , Run ==python train.py num== to generate the first generation model
6. From cell (# # # -- -- -- -- -- - # # # #) to run, to generate the next generation of the data.
7. ==num+=1==, Loop 5 ~ 6 until num = 6. In general, it takes 5 times to train the model to get the final ==submitted model==, and 4 times to get the final submitted ==data.npy==
