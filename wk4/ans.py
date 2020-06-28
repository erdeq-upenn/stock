#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 21:41:49 2020

@author: Dequan
"""

class Date(object):
   day = 0
   month = 0
   year = 0

   def __init__(self, year=0, month=0, day=0):
       self.day = day
       self.month = month
       self.year = year

   @classmethod
   def from_string(cls, date_as_string): # 类方法，我们不用通过实例化类就能访问的方法. 有cls，约定参数，会更改类的结果
       year, month, day = date_as_string.split('-')
       date = cls(year, month, day)
       return date

   @staticmethod
   def is_date_valid(date_as_string): # 这里只有一个参数，不需要self，也不会更改类的结果
       """
      用来校验日期的格式是否正确
      True or false, 
       """
       year, month, day = date_as_string.split('-')
       return int(year) <= 3999 and int(month) <= 12 and int(day) <= 31



if __name__ == '__main__':
    
    date1 = Date.from_string('2012-05-10')
    print(date1)
    print(date1.year, date1.month, date1.day)
    is_date = Date.is_date_valid('2012-09-32') # 格式正确 返回True
    print(is_date)