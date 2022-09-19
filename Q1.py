#size of hash table is 10 bits
import random;
from sklearn.utils import murmurhash3_32;
import mmh3;
import numpy as np;
import seaborn as sns;
import matplotlib.pyplot as plt;
import pandas as pd;
import sys;
#Used below code to generate a,b,c,d

random.seed(500);

start=1;
end=1048573;
a=random.randint(start, end);
b=random.randint(start, end) ;
c=random.randint(start, end);
d=random.randint(start, end);
P=1048573;

# print(a,b,c,d)


# a=449874;
# b=924891;
# c=49399;
# d=439425;



#2 Universal
def Hash1(x):
    return ((a*x+b) % P)%1024;

#3 Universal
def Hash2(x):
    return ((a*(x**2)+b*x+c) % P)%1024;

#4 Universal
def Hash3(x):
    return ((a*(x**3)+b*(x**2)+c*x+d) % P)%1024;

#5 Murmur
def Hash4(x):
    return murmurhash3_32(x,1, positive=True) % 1024;

#Generating 5000 random 31 bit numbers
temp=[random.getrandbits(31) | 1 for i in range (5000) ];
# print(temp);
# t1=[(31-len(bin(x)[2:])) for x in temp];
# print(t1);
temp=[x<<(31-len(bin(x)[2:])) for x in temp];
# print(temp);
# t2=[len(bin(x)[2:]) for x in temp];
# print(t2);
# print(temp)
print(temp[:5]);


#Arrays for storing the output changes for each input bit for all the 4 hash functions.
probability1=np.zeros([31,10]);
probability2=np.zeros([31,10]);
probability3=np.zeros([31,10]);
probability4=np.zeros([31,10]);

#Finding the changes in output bits for 2 Universal Hash Function
for num in temp:
    old=Hash1(num);
    for i in range (31):
        new=Hash1((1<<i) ^ num);
        binary=old ^ new;
        for k in range (10):
            probability1[i][9-k]+= 1 if (binary & (1<<k))>=1 else 0 ;
    
#Finding the changes in output bits for 3 Universal Hash Function
for num in temp:
    old=Hash2(num);
    for i in range (31):
        new=Hash2(((1<<i) ^ num));
        binary=old ^ new;
        for k in range (10):
            probability2[i][9-k]+= 1 if (binary & (1<<k))>=1 else 0 ;

#Finding the changes in output bits for 4 Universal Hash Function
for num in temp:
    old=Hash3(num);
    for i in range (31):
        new=Hash3(((1<<i) ^ num));
        binary=old ^ new;
        for k in range (10):
            probability3[i][9-k]+= 1 if (binary & (1<<k))>=1 else 0 ;

#Finding the changes in output bits for Murmur Hash Function
for num in temp:
    old=Hash4(num);
    for i in range (31):
        new=Hash4(((1<<i) ^ num));
        binary=old ^ new;
        for k in range (10):
            probability4[i][9-k]+= 1 if (binary & (1<<k))>=1 else 0 ;

#Calculating the probability
probability1=probability1/5000;
probability2=probability2/5000;
probability3=probability3/5000;
probability4=probability4/5000;
x=np.linspace(0,31,num=31);

#Plotting Starts
dfx = pd.DataFrame(x, columns = ['Input Bits']);
dfy2=pd.DataFrame(probability1,columns=['0','1','2','3','4','5','6','7','8','9']);
dfy3=pd.DataFrame(probability2,columns=['0','1','2','3','4','5','6','7','8','9']);
dfy4=pd.DataFrame(probability3,columns=['0','1','2','3','4','5','6','7','8','9']);
dfym=pd.DataFrame(probability4,columns=['0','1','2','3','4','5','6','7','8','9']);

#2 universal hash function
plt.figure(figsize=(10.0,10.0));
plt.title("2 universal hash function");
plt.xlabel('Input bits',size=31);
plt.ylabel('Output bits',size=10);
plt.plot();
sns.heatmap(dfy2.T,fmt=".1f",annot=True,linewidths=2,square=True,cmap='twilight');

#3 universal hash function
plt.figure(figsize=(10.0,10.0));
plt.title("3 universal hash function");
plt.xlabel('Input bits',size=31);
plt.ylabel('Output bits',size=10)
plt.plot();
sns.heatmap(dfy3.T,fmt=".1f",annot=True,linewidths=2,square=True,cmap='twilight');

#4 universal hash function
plt.figure(figsize=(10.0,10.0));
plt.title("4 universal hash function");
plt.xlabel('Input bits',size=31);
plt.ylabel('Output bits',size=10)
plt.plot();
sns.heatmap(dfy4.T,fmt=".1f",annot=True,linewidths=2,square=True,cmap='twilight', vmin=0.5, vmax=0.5);

#Murmur hash function
plt.figure(figsize=(10.0,10.0));
plt.title("Murmur hash function");
plt.xlabel('Input bits',size=31);
plt.ylabel('Output bits',size=10)
plt.plot();
sns.heatmap(dfym.T,fmt=".1f",annot=True,linewidths=2,square=True,cmap='twilight', vmin=0.5, vmax=0.5);

plt.show()
