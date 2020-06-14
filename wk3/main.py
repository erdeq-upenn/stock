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
    # answer 
    content = '北京欢迎你为你开天辟地…………'
    count = 0
    while count<=10:
        # 清理屏幕上的输出
#        os.system('cls')  # os.system('clear') # for .ipynb format 
        print(content)
        # 休眠200毫秒
        time.sleep(0.2)
        time.sleep(0.2)
        content = content[1:] + content[0]
        count+=1
        # your code 

def ploygon(n):
    t = turtle.Pen()                           # 获得画笔
    t.pensize(2)     
    d = 400
    for _ in range(8):
        for i in range(n):
            t.forward(d*sin(pi/n))
            t.left(360/n)
    
#        print(d)
        # t.left(360/n)
        x =0.5
        t.forward(x*d*sin(pi/n))
        t.left(np.arctan(x/(1-x))*180/pi)
        d = d * np.sqrt(x**2+(1-x)**2)
    
    t.hideturtle()                            # 隐藏光标
    ts = turtle.getscreen()                   # 保存图片
    ts.getcanvas().postscript(file="img2.eps")
    turtle.exitonclick()                      # 单击画布退出

#--------------------------------------------
def main():
    ploygon(4)

#--------------------------------------------
N_test = 4
def test_var(N_test):
    global out 
    out = N_test**2 
    return out

if __name__ == '__main__':
#    print(test_var(4))
#    print(out)
    from example import save_fig,run_linear,run_poly,run_sgd
    run_sgd()
