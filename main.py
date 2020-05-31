#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 09:42:53 2020

@author: Dequan
"""


"""
This is a comment
"""


def myfun(x):
    '''
    return x**3
    '''
    
    y = x**4-x**3-x**2

    return y

def myloop(n):
    # n is int. input
#    assert type(n)=='int'
    sum =0
    for x in range(n+1):
        sum +=x
#    print(sum)

    return ('The total sum is %s' %sum)

def ninetable():
   for i in range(1,10):
       for j in range(1,i+1):
           print('%d*%d=%d'%(i,j,i*j),end='\t')
       print()

def mycond(n):
    # if n > 10, print n
    if n>=10:
        return n
    elif n>0:
        return n**2
    else:
        return 'Negative'

#--------------------------------------------
def main():
#    for i in range(10):
#        print(myfun(i))
#    print(myloop(10))
#    ninetable()
    print(mycond(-2))
#--------------------------------------------

if __name__ == '__main__':
    main()
