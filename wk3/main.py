#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:55:41 2020

@author: Dequan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 09:42:53 2020

@author: Dequan
"""


"""
This is a comment
"""
from math import sin, pi, cos                # 正弦函数和圆周率
import turtle
import numpy as np
import os 
import time 

def text():
    content = '北京欢迎你为你开天辟地…………'
    count = 0
    while count<=10:
        # 清理屏幕上的输出
        os.system('cls')  # os.system('clear')
        print(content)
        # 休眠200毫秒
        time.sleep(0.2)
        # your code 
        # left blank intenionally 
        # your code 

def varilen(x,*argv):
    print(x,*argv)
    
def myfun(t):
    '''
    test
    '''
    t.forward(100)
    t.hideturtle()                            # 隐藏光标
#    ts = turtle.getscreen()                   # 保存图片
#    ts.getcanvas().postscript(file="img2.eps")
    turtle.exitonclick()   
#--------------------------------------------
def main():
    t = turtle.Pen()                           # 获得画笔
    t.pensize(3)  
    for _ in range(4):
        t.forward(100)
        t.left(90)
    turtle.exitonclick()  
    text()
#    varilen(5,myDict)
#--------------------------------------------

if __name__ == '__main__':
    main()
