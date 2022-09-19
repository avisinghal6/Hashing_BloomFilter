import csv
import pandas as pd
import random;
from sklearn.utils import murmurhash3_32;
import mmh3;
import numpy as np;
from bitarray import bitarray
import math;
import string;
import matplotlib.pyplot as plt;
import sys;

random.seed(2315); 
data = pd.read_csv('/Users/avisinghal/Desktop/Rice Sem1/COMP 580/HW1/user-ct-test-collection-01-1.txt', sep="\t")
urllist = data.ClickURL.dropna().unique();
class BloomFilter():

    def __init__(self,n,r):
        self.count=0;
        self.hashseed=random.randint(1,500);
        self.A=bitarray(r);
        self.A.setall(0); # initializing bitarray to 0.
        self.k=math.ceil(0.7*r/n);
        self.hash=[];
        #print(self.k);
        for i in range(self.k):
            self.hash.append(self.hashfunc(r));

        self.membership=urllist;
        self.random_membership=[random.choice(self.membership) for i in range (1000)];
        
        #print(self.random_membership);
        for i in self.membership :
            for j in range(self.k): 
                self.A[self.hash[j](i)]=1;



    def hashfunc(self,m):
        a=self.hashseed+self.count;
        self.count=self.count+1;
        def murmur(x):
            #print(a);
            return murmurhash3_32(x,a) % m ;
        

        return murmur;

    def test(self,x):
        ans=0;
        for i in range(self.k):
            if self.A[self.hash[i](x)]==1 :
                ans+=1;


        if ans==self.k :
            return 1;
        else :
            return 0;


#print(len(urllist));

testset=[''.join(random.choices(string.ascii_lowercase, k=7)) for i in range (1000)];

# start from 19 bits

x=[];
y=[];
for i in range(19,25):
    bf_obj=BloomFilter(377871,1<<i );
    fp=0;
    for k in testset:
        fp+=bf_obj.test(k);

    fp/=1000;
    x.append(sys.getsizeof(bf_obj.A)+sys.getsizeof(bf_obj.hash));
    print("Size of BloomFilter object for  " ,i,"bits( R=",1<<i,") ","Memory Size :",sys.getsizeof(bf_obj.A)+sys.getsizeof(bf_obj.hash));
    y.append(fp);
    print("{0:.8f}".format(fp));


plt.plot(x, y);
plt.xlabel('Memory Size');
plt.ylabel('False Positive Rate');
plt.show()


py_dict= dict();

for i in urllist:
    py_dict[i]=1;

print("Size of Python Hash table for storing same urlist",sys.getsizeof(py_dict));


