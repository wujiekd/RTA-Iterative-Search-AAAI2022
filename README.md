## How to run

1. Obtain all Cifar10 data from the official website to the folder ==CIFAR-10-FANs-PY==

2. Create two empty folders.  ==new_data==and ==train_model== 
3. Run ==gen_corr_data.ipynb== to generate corruption data

4. Run ==final_gen_data_iter. Ipynb== to cell (# # # -- -- -- -- -- - # # # #) to stop, to generate the first generation of the data.
5. ==num = 1== , Run ==python train.py num== to generate the first generation model
6. From cell (# # # -- -- -- -- -- - # # # #) to run, to generate the next generation of the data.
7. ==num+=1==, Loop 5 ~ 6 until num = 6. In general, it takes 5 times to train the model to get the final ==submitted model==, and 4 times to get the final submitted ==data.npy==