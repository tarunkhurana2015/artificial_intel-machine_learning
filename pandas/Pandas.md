# Pandas

Pandas is a Python library used for working with data sets.
It has functions for analyzing, cleaning, exploring and manipulating data.

## Installation of Pandas

```python
pip install pandas
```

Once Pandas is installed, import it in your applications by adding the import keyword

```python
import pandas as pd
```

Checking Pandas Version
`The version string is stored under __version__ attribute.`

## Pandas Series

A Pandas Series is like a column in a table.
It is a one-dimensional array holding data of any type.

## Labels

If nothing else is specified, the values are labeled with their index number. First value has index 0, second value has index 1 etc.
This label can be used to access a specified value.

### Create Labels

With the index argument, you can name your own labels.

### Key/Value Objects as Series

You can also use a key/value object, like a dictionary, when creating a Series.

## DataFrames

Data sets in Pandas are usually multi-dimensional tables, called DataFrames.

Series is like a column, a DataFrame is the whole table.

### Locate Row

As you can see from the result above, the DataFrame is like a table with rows and columns.

Pandas use the loc attribute to return one or more specified row(s)

### Named Indexes

With the index argument, you can name your own indexes.

### Locate Named Indexes

Use the named index in the loc attribute to return the specified row(s).

### Load Files Into a DataFrame

If your data sets are stored in a file, Pandas can load them into a DataFrame.

```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df)
```

## Read JSON

Big data sets are often stored, or extracted as JSON.

JSON is plain text, but has the format of an object, and is well known in the world of programming, including Pandas.

```python
import pandas as pd
df = pd.read_json('data.json')
print(df.to_string())
```

### Viewing the Data

One of the most used method for getting a quick overview of the DataFrame, is the `head()` method.

The `head()` method returns the headers and a specified number of rows, starting from the top.

```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head(10))
```

There is also a `tail(`) method for viewing the last rows of the DataFrame.

The `tail()` method returns the headers and a specified number of rows, starting from the bottom.

```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.tail())
```

### Info About the Data

The DataFrames object has a method called `info()`, that gives you more information about the data set.

```python
print(df.info())
```

## Preprocessing using Pandas:

### Replace Empty Values

- Remove rows that contain empty cells/ Replace Empty Values
  - The `fillna()` method allows us to replace empty cells with a value
- Replace Using Mean, Median, or Mode
  - Pandas uses the `mean()` ,`median()` and `mode()` methods to calculate the respective values for a specified column
    - Mean = the average value (the sum of all values divided by number of values).
      - The mean() function is also useful for data exploration, which is the process of discovering patterns, relationships, and insights in data. By calculating the mean of a certain column, we can get an idea of the typical value for that column and how it relates to other variables in our dataset. We can also use the mean() function to compare different subsets of our data and identify any trends or patterns.
    - Median = the value in the middle, after you have sorted all values ascending.
    - Mode = the value that appears most frequently.

### Clean Data

Cells with data of wrong format can make it difficult, or even impossible, to analyze data.

To fix it, you have two options:

- remove the rows
- convert all cells in the columns into the same format.

#### Removing Rows

The result from the converting in the example above gave us a NaT value, which can be handled as a NULL value, and we can remove the row by using the `dropna()` method.

#### Removing Duplicates

To discover duplicates, we can use the `duplicated()` method. The `duplicated()` method returns a Boolean values for each row.

### Data Correlations

A great aspect of the Pandas module is the `corr()` method.

The `corr()` method calculates the relationship between each column in your data set.

The Result of the `corr()` method is a table with a lot of numbers that represents how well the relationship is between two columns.

The number varies from `-1 to 1`.

`1` means that there is a 1 to 1 relationship (a perfect correlation), and for this data set, each time a value went up in the first column, the other one went up as well.

`0.9` is also a good relationship, and if you increase one value, the other will probably increase as well.

`-0.9` would be just as good relationship as 0.9, but if you increase one value, the other will probably go down.

`0.2` means NOT a good relationship, meaning that if one value goes up does not mean that the other will.

### Plotting

Pandas uses the `plot()` method to create diagrams.

We can use `Pyplot`, a submodule of the `Matplotlib` library to visualize the diagram on the screen.

#### Scatter Plot

Specify that you want a scatter plot with the kind argument:

kind = `scatter` - A scatter plot needs an x- and a y-axis.

#### Histogram

Use the kind argument to specify that you want a histogram:

kind = 'hist'

A histogram needs only one column.

A histogram shows us the frequency of each interval
