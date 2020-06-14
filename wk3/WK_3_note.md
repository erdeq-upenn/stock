# 斐波那契数列
```
a = 0
b = 1
for _ in range(20):
    a, b = b, a + b
    print(a, end=' ')
```
# code for tensorflow
```
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
```


# normal equation
[here](http://mlwiki.org/index.php/Normal_Equation)
[here](https://www.geeksforgeeks.org/ml-normal-equation-in-linear-regression/)

＃ 类方法
```
class Date(object):
   day = 0
   month = 0
   year = 0

   def __init__(self, year=0, month=0, day=0):
      # your code

   @classmethod
   def from_string(##): # 类方法，我们不用通过实例化类就能访问的方法. 有cls，约定参数，会更改类的结果
       # your code , parse '2020-01-01' to year, month, date
       return date

   @staticmethod
   def is_date_valid(date_as_string): # 这里只有一个参数，不需要self，也不会更改类的结果
       """
      用来校验日期的格式是否正确
       """
       year, month, day = date_as_string.split(#your code #)
       # year <=3999, 0< month <=12,0<day<31
       # additional if you can check Feb 29 for leap year
       return # your code

# check code here: date1 = Date.from_string('2012-05-10')
print(date1.year, date1.month, date1.day)
is_date = Date.is_date_valid('2012-09-18') # 格式正确 返回True
```
