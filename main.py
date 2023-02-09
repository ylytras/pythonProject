import numpy as np
from numpy import ndarray

'''arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))

arr1 = np.array(10)
print(arr1)

arr2 = np.array([[1, 2], [3, 4]])
print(arr2)

arr3 = np.array([[[1, 2], [3,4]], [[5, 6], [7, 8]]])
print(arr3)

print(arr1.ndim,arr2.ndim, arr3.ndim)

arr = np.array([1, 2, 3, 4, 5])
print(arr[0])

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[0, 0])
print(arr[0, -1])

arr = np.array([10, 20, 30, 40, 50])
print(arr[1:4])
print(arr[2:])
print(arr[:4])

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8,])
print(arr[0:7:2])

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[1, 0:2])

arr = np.array([1, 2, 3, 4, 5])
copy = arr.copy()
copy[0] = 24
print(arr)
print(copy)

arr = np.array([1, 2, 3, 4, 5])
view = arr.view()
view[0] = 24
print(arr)
print(view)

arr = np.array([[1, 2, 3], [4, 5, 6]])
copy = arr.copy()
print(arr)
print()
print(arr.shape)

arr = np.array([1, 2, 3, 4, 5, 6,])
view = arr.view()
print(arr)
print()
print(arr.reshape(2, 3))'''

'''arr = np.array([1, 2, 3, 4, 5, 6])
arrList = np.array_split(arr,3)
for arr in arrList:
    print(arr)

arr = np.array([1, 2, 3, 2, 4, 2])
print(np.where(arr == 2))

arr = np.array([1, 2, 3, 4, 5, 6])
print(np.where(arr % 2 == 0))

arr = np.array([3, 1, 2, 6, 4, 5])
print(np.sort(arr))

arr = np.array([True, False, False, True])
print(np.sort(arr))

arr = np.array(["pasta", "beans", "cake"])
print(np.sort(arr))

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([1, 2, 3, 4])
print(np.add(arr1, arr2))

arr1 = np.array([10, 20, 30, 40])
arr2 = np.array([1, 3, 4, 5])
print(np.subtract(arr1, arr2))

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([1, 2, 3, 4])
print(np.multiply(arr1, arr2))

arr1 = np.array([10, 20, 30, 40])
arr2 = np.array([1, 2, 3, 4])
print(np.divide(arr1, arr2))

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([1, 2, 3, 4])
print(np.power(arr1, arr2))

arr1 = np.array([10, 10, 10, 10])
arr2 = np.array([1, 2, 3, 4])
arr3 = np.array([-1, 2, -3, 4])
print(np.mod(arr1, arr2))
print(np.absolute(arr3))

arr = np.array([1.23, 3.45, 6.78])
print(np.trunc(arr))
print(np.fix(arr))

arr = np.array([1.23, 3.45, 6.78])
print(np.around(arr, 1))

arr = np.array([1.2345, 6.789])
print(np.floor(arr))
print(np.ceil(arr))

arr = np.array([1, 2, 3, 4, 5])
print(np.log(arr))
print(np.log2(arr))
print(np.log10(arr))

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
print(np.sum([arr1, arr2]))

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
print(np.sum([arr1, arr2], axis=1))

arr = np.array([1, 2, 3])
print(np.cumsum(arr))

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
print(np.prod([arr1, arr2]))

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
print(np.prod([arr1, arr2], axis=1))

arr = np.array([1, 2, 3])
print(np.cumprod(arr))

arr = np.array([1, 2, 3, 4, 5])
print(np.lcm.reduce(arr))
print(np.gcd.reduce(arr))

arr = np.array([np.pi/2, np.pi/3, np.pi/4])
print(np.around(np.sin(arr), 8))
print(np.around(np.cos(arr), 8))

arr = np.array([90, 180, 270])
arr = np.deg2rad(arr)
print(arr)
arr = np.rad2deg(arr)
print(arr)

print(np.hypot(3, 4))

import pandas as pd

x = [23, 48, 19]
my_first_series = pd.Series(x)
print(my_first_series)
import pandas as pd
data = {
    "students": ["Emma", "John", "Paul"],
    "grades": [12, 18, 17]
    }
my_first_dataframe = pd.DataFrame(data)
print(my_first_dataframe)

import pandas as pd
data = {
    "students": ["Emma", "John", "Paul"],
    "grades": [12, 18, 17]
}
my_first_dataframe = pd.DataFrame(data)
print(my_first_dataframe["students"])

import pandas as pd
data = {
    "students": ["Emma", "John", "Paul"],
    "grades": [12, 18, 17]
}
my_first_dataframe = pd.DataFrame(data, index=["a", "b", "c"])
first_row = my_first_dataframe.loc["b"]
print(first_row)

import pandas as pd
data = {
    "students": ["Emma", "John", "Paul"],
    "grades": [12, 18, 17]
}
my_first_dataframe = pd.DataFrame(data, index=["a", "b", "c"])
second_row = my_first_dataframe.iloc[2]
print(second_row)

import pandas as pd
import numpy as np
data = {
    "students": ["Emma", "John", np.nan, "Bob"],
    "grades": [12, np.nan, 17, 18]
}
my_first_df = pd.DataFrame(data, index=["a", "b", "c", "d"])
print(my_first_df.isnull())

import pandas as pd
import numpy as np
data = {
    "students": ["Emma", "John", np.nan, "Bob"],
    "grades": [12, np.nan, 17, 18]
}
my_first_df = pd.DataFrame(data, index=["a", "b", "c", "d"])
my_first_df["students"].fillna("No Name", inplace=True)
my_first_df["grades"].fillna("No Grade", inplace=True)
print(my_first_df)

import pandas as pd
import numpy as np
data = {
    "students": ["Emma", "John", np.nan, "Bob"],
    "grades": [12, np.nan, 18, 17]
}
my_first_df = pd.DataFrame(data, index= ["a", "b", "c", "d"])
my_first_df["students"].fillna("No name", inplace=True)
my_first_df["grades"].fillna("No grade", inplace=True)
df2 = my_first_df.replace(to_replace="Bob", value="Alice")
print(df2)

import pandas as pd
import numpy as np
data = {
    "students": ["Emma", "John", "Mary", 'Bob'],
    "grades": [12, np.nan, 18, np.nan]
}
my_first_df = pd.DataFrame(data, index=["a", "b", "c", "d"])
df = my_first_df.interpolate(method="linear", limit_direction="forward")
print(df)

import pandas as pd
import numpy as np
data = {
    "students": ["Emma", "John", "Mary", 'Bob'],
    "grades": [12, np.nan, 18, np.nan]
}
my_first_df = pd.DataFrame(data, index=["a", "b", 'c', 'd'])
my_first_df.dropna(inplace=True)
print(my_first_df)

import pandas as pd
s = pd.Series(['workeary', 'elearning', 'python'])
for index, value in s.items():
    print(f"Index: {index}, 'Value: {value}")

import pandas as pd
data = {
    'students': ['Emma', 'John'],
    'grades': [12, 19.8]
}
my_first_df = pd.DataFrame(data, index=['a', 'b'])
for i,j in my_first_df.iterrows():
    print(i,j)
    print()

import pandas as pd
data = {
    'students': ['Emma', 'John'],
    'grades': [12, 19.8]
}
my_first_df = pd.DataFrame(data, index=['a', 'b'])
columns = list(my_first_df)
for i in columns:
    print(my_first_df[i][1])

import pandas as pd
df = pd.read_csv('finance_liquor_sales.csv')
print(df.head())
print(df.tail())
print(df.info())
print(df.shape)

import pandas as pd
df = pd.read_csv('finance_liquor_sales.csv')
mean = df.mean(numeric_only=True)
print(mean)

import pandas as pd
df = pd.read_csv('finance_liquor_sales.csv')
median = df.median(numeric_only=True)
print(median)

import pandas as pd
df = pd.read_csv('finance_liquor_sales.csv')
max = df.max(numeric_only=True)
print(max)

import pandas as pd
df = pd.read_csv('finance_liquor_sales.csv')
summary = df.describe()
print(summary)

import pandas as pd
df = pd.read_csv('finance_liquor_sales.csv')
cn = df.groupby('category_name')
print(cn.first())

import pandas as pd
df = pd.read_csv('finance_liquor_sales.csv')
cnc = df.groupby(['category_name', 'city'])
print(cnc.first())

import pandas as pd
import numpy as np
df = pd.read_csv('finance_liquor_sales.csv')
cn = df.groupby('category_name')
print(cn.aggregate(np.sum))

import pandas as pd
df = pd.read_csv("finance_liquor_sales.csv")
cn2 = df.groupby(['category_name', 'city'])
print(cn2.agg({'bottles_sold': 'sum', 'sale_dollars': 'mean'}))

import pandas as pd
df = pd.read_csv('finance_liquor_sales.csv')
cng = df.groupby('vendor_name')
print(cng.filter(lambda x len(x) >= 20))'''

'''import pandas as pd
df = pd.read_csv("finance_liquor_sales.csv")
ng = df.groupby('vendor_name')
print(ng.filter(lambda x: len(x) >= 20))'''
import pandas as pd

'''d1 = dict(Name=['Mary', 'John', 'Alice', 'Bob'],
          Age=[27, 24, 22, 32],
          Position=['Data Analyst', 'Trainee', 'QA Tester', 'IT'])
d2 = dict(Name=['Steve', 'Tom', 'Jenny', 'Nick'],
          Age=[35, 28, 41, 52],
          Position=['IT', 'Data Analyst', 'Consultant', 'IT'])
df1 = pd.DataFrame(d1, index=[0, 1, 2, 3])
df2 = pd.DataFrame(d2, index=[4, 5, 6, 7])
result = pd.concat([df1, df2])
print(result)'''

'''d1 = {'key': ['a', 'b', 'c', 'd'], 'Name': ['Mary', 'John', 'Alice', 'Bob']}
d2 = {'key': ['a', 'b', 'c', 'd'], 'Age': [27, 32, 45, 53]}
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)
result = pd.merge(df1, df2, on='key')
print(result)'''

'''d1 = dict(Name=['Mary', 'John', 'Alice', 'Bob'], Age=[27, 32, 45, 53])
d2 = {'Position': ['Data Analyst', 'Trainee', 'QA Tester', 'IT'],
      'Years of Experience':[5, 1, 10, 3]}
df1 = pd.DataFrame(d1, index=[0, 1, 2, 3])
df2 = pd.DataFrame(d2, index=[0, 2, 3, 4])
result = df1.join(df2, how= 'inner')
print(result)

L = [5, 10, 15, 20, 25]
LS = pd.Series(L)
print(LS)'''

'''import pandas as pd

d = {'col1': [1, 2, 3, 4, 7, 11],
     'col2': [4, 5, 6, 9, 5, 0],
     'col3': [7, 5, 8, 12, 1, 11]}
df = pd.DataFrame(d)
result = df.iloc[:, 0]
print('first column as series')
print(result)
print(type(result))

df = pd.read_csv('text.csv')
print(df.head(20))

df = pd.read_csv('text.csv')
for i, j in df.iterrows():
    print(i,j)


import pandas as pd
import numpy as np
data = pd.read_csv('1.supermarket.csv')
x = data.groupby('item_name')
x = x.sum()
print(x.head(1))'''

import matplotlib.pyplot as plt

'''plt.plot([0, 10], [0, 300], 'o')
plt.show()
plt.plot([0, 2, 4, 6, 8, 10], [2, 1, 5, 4, 2, 15])
plt.show()

plt.plot([0, 3, 5],[0, 5, 9], ls='dotted', marker='o')
plt.show()

plt.plot([0, 10], [0, 300], 'o')
plt.title('Title')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid()
plt.show()

plt.subplot(2, 1, 1)
plt.plot([0, 2, 4, 6, 8, 10], [3, 8, 1, 10, 5, 12])
plt.subplot(2, 1, 2)
plt.plot([0, 10],[0, 300])
plt.show()'''

import matplotlib.pyplot as plt
import numpy as np

'''x = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])
y = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
plt.scatter(x, y)
x = np.array([100, 105, 84, 105, 90, 99, 90, 95, 94, 100, 79, 112, 91, 80, 85])
y = np.array([2, 2, 8, 1, 15, 8, 12, 9, 7, 3, 11, 4, 7, 14, 12])
plt.scatter(x, y)
plt.show()'''

'''x = np.array(['a', 'b', 'c', 'd'])
y = np.array([2, 4, 6, 8])
plt.bar(x,y)
plt.show()

my_labels = np.array(['Tomatoes', 'Lemons', 'Sausages', 'Bacon'])
x = np.array([15, 25, 25, 35])
plt.pie(x, labels=my_labels)
plt.legend()
plt.show()

age = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
cardiac_cases = [5, 15, 20, 40, 55, 55, 70, 80, 90, 95]
survival_rate = [99, 99, 90, 90, 80, 75, 60, 50, 30, 25]
plt.xlabel('Age')
plt.ylabel('Percentage')
plt.plot(age, cardiac_cases, color='black',linewidth=2, label="cardiac_cases", marker="o", markerfacecolor="red", markersize=12)
plt.plot(age, survival_rate, color='yellow',linewidth=3, label="survival_rate", marker="o", markerfacecolor="green", markersize=12)
plt.legend(loc="lower right", ncol=1)
plt.show()

products = np.array([
    ["Orange", "Apple"],
    ["Beef", "Chicken"],
    ["Candy", "Cocolate"],
    ["Fish", "Bread"],
    ["Eggs", "Bacon"]])
random = np.random.randint(2, size=5)
choices = []
counter = 0
for product in products:
    choices.append(product[random[counter]])
    counter +=1
print(choices)
percentages = []
for i in range(4):
    percentages.append(np.random.randint(25))
percentages.append(100 - np.sum(percentages))
print(percentages)
plt.pie(percentages, labels=choices)
plt.legend(loc='lower right', ncol=1)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('1.supermarket.csv')
q = data.groupby('item_name').quantity.sum()
plt.bar(q.index, q, color=['orange', 'purple', 'yellow', 'red', 'green', 'blue', 'cyan'])
plt.xlabel('Items')
plt.xticks(rotation=6)
plt.title('Most ordered Supermarket\'s items')
plt.ylabel('Number of items ordered')
plt.show()
import pandas as pd
import requests
from bs4 import BeautifulSoup
url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, "html.parser")
tables = s. find_all('table')
print(tables)

import requests
from bs4 import BeautifulSoup
url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, "html.parser")
my_table = s.find('table', class_='wikitable sortable plainrowheaders')
table_links = my_table.find
print(tables)

import requests
from bs4 import BeautifulSoup
url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, "html.parser")
my_table = s.find_all('table', class_='wikitable sortable plainrowheaders')
table_links = my_table.find()
actors = []
for links in table_links:
      actors.append(links.get('title'))
print(actors)'''

