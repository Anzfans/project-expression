#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Author: Zhenghao Li
# Email: lizhenghao@shanghaitech.edu.cn
# Institute: SIST
# Date: 2024-12-10

import torch
import my_net


batch_size=60

model, lossfun, optimizer = my_net.classify.makeEmotionNet(False)
PATH = './face_expression(Batch_size=60).pth'
model.load_state_dict(torch.load(PATH, weights_only=True))
test_loader, classes = my_net.utility.loadTest("./images/", batch_size)

X,y = next(iter(test_loader))

model.eval()
## Test in one batch
with torch.no_grad():
    yHat = model(X)

    ##Step 1 Obtain predicted labels
    new_labels= y
    max_yhat, predicted_labels=torch.max(yHat,dim=1)
    y=y.tolist()
    predicted_labels=predicted_labels.tolist()



    

    ##Show first 32 predicted labels
    my_net.utility.imshow_with_labels(X[:batch_size], new_labels[:batch_size], classes)

    #Step 2
    ##Calculate the accuracy for each category prediction, as well as the overall accuracy
    #Print them to the screen.
    ## "happy:xx.xx%, neutral:xx.xx%, sad:xx.xx%, total:xx.xx%"
def accuracy(label):
    count1=0
    count2=0
    for i in predicted_labels:
        if i==label:
            count1+=1
    for i in range(len(y)):
        if y[i]==predicted_labels[i] and y[i]==label:
            count2+=1
    
    return count2/count1
total1=sum( x==y for x, y in zip(y, predicted_labels))/len(y)

print(f'happy:{accuracy(0)*100:.2f}%, neutral:{accuracy(1)*100:.2f}%, sad:{accuracy(2)*100:.2f}%, total:{total1*100:.2f}%')


 


    #Step 3
    ##Calculate the recall for each category prediction, as well as the overall accuracy
    #Print them to the screen.
    ## "happy:xx.xx%, neutral:xx.xx%, sad:xx.xx%, total:xx.xx%"
def recall(label):
    count1=0
    count2=0
    for i in y:
        if i==label:
            count1+=1
    for i in range(len(y)):
        if y[i]==predicted_labels[i] and y[i]==label:
            count2+=1
    
    return count2/count1
total2=(y.count(0)/len(y))*recall(0)+(y.count(1)/len(y))*recall(1)+(y.count(2)/len(y))*recall(2)
   
print(f'happy:{recall(0)*100:.2f}%, neutral:{recall(1)*100:.2f}%, sad:{recall(2)*100:.2f}%, total:{total2*100:.2f}%')
## Get the accuracy and recall in full dataset
##Step 4
label={"accuracy":[0,0,0,0], "recall":[0,0,0,0]}
total_sample=len(test_loader.dataset)//batch_size
with torch.no_grad():
    for X,y in test_loader:
        yHat = model(X)
        max_yhat, predicted_labels=torch.max(yHat,dim=1)
        y=y.tolist()
        predicted_labels=predicted_labels.tolist()
        for i in range(len(label['accuracy'])-1):
            label['accuracy'][i]+=accuracy(i)
            label['recall'][i]+=recall(i)
        label['accuracy'][3]+=sum(x==y for x, y in zip(y, predicted_labels))/len(y)
        label['recall'][3]+=(y.count(0)/len(y))*recall(0)+(y.count(1)/len(y))*recall(1)+(y.count(2)/len(y))*recall(2)


print(f'happy:{label['accuracy'][0]/total_sample*100:.2f}%, neutral:{label['accuracy'][1]/total_sample*100:.2f}%, sad:{label['accuracy'][2]/total_sample*100:.2f}%, total:{label['accuracy'][3]/total_sample*100:.2f}%')
print(f'happy:{label['recall'][0]/total_sample*100:.2f}%, neutral:{label['recall'][1]/total_sample*100:.2f}%, sad:{label['recall'][2]/total_sample*100:.2f}%, total:{label['recall'][3]/total_sample*100:.2f}%')

    
