import numpy as np
import matplotlib.pyplot as plt 
from collections import Counter

#Gini index
#Gini index gives us an idea of how well the data labels are split 
#Worst score - 50 / 50 split 
#Best score - all one label - 0
#This score is then weighted by the number of samples in each group 

#Example: 
#Animal	Size	Color	Category
#A	Small	Red	Yes
#B	Large	Red	No
#C	Small	Blue	Yes
#D	Large	Blue	No
#E	Small	Red	No
#F	Large	Blue	Yes
#G	Small	Blue	No
#H	Large	Red	Yes

#Label is Category (Yes, No)
#Gini index for Animal = Small 
#Small - Yes: 4, No: 2
# = 1 - (4/6)^2 - (2/6)^2 = .44 

def gini_index(groups, classes):
    
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        class_counts = Counter([row[-1] for row in group])  # Count occurrences
        gini_group = 1.0
        for class_val in classes:
            p = class_counts[class_val] / size
            print(f"Class {class_val} count: {class_counts[class_val]}, Probability: {p:.3f}")
            gini_group -= p ** 2

        print(f"Gini Index for group: {gini_group:.3f}")
        print(f"Gini Group Contribution: {gini_group:.3f} * {size / n_instances:.3f}")
        gini += gini_group * (size / n_instances)
    
    return gini



# Example dataset: [Color, Size, Class]
# Color: 0 - Red, 1 - Orange
# Size: 0 - Small, 1 - Large
# Class: 0 - Apple, 1 - Orange
dataset = [
    [0, 0, 0],  # Red, Small, Apple
    [1, 1, 1],  # Orange, Large, Orange
    [0, 1, 0],  # Red, Large, Apple
    [1, 0, 1],  # Orange, Small, Orange
    [0, 1, 1],  # Red, Large, Orange
    [1, 0, 0],  # Orange, Small, Apple
]

# Split the dataset into groups for demonstration
group1 = dataset[:3]
group2 = dataset[3:]

# List of unique class labels
classes = [0, 1]

# Calculate Gini Index for the split
#gini_group1 = gini_index([group1], classes)
#gini_group2 = gini_index([group2], classes)

#Splitting the data 
#We need to split data based on an attribute or feature in order to get its gini values 
def test_split(index, value, dataset):
 left, right = list(), list()
 for row in dataset:
    if row[index] < value:
        left.append(row)
    else:
        right.append(row)
 return left, right

dataset = [
    [2.3, 1.5, 0],
    [3.7, 2.2, 1],
    [1.1, 0.8, 0],
    [4.0, 2.8, 1],
    [2.8, 1.9, 1]
]

#print(test_split(0,3.0, dataset))
#([[2.3, 1.5, 0], [1.1, 0.8, 0], [2.8, 1.9, 1]], [[3.7, 2.2, 1], [4.0, 2.8, 1]])

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None

    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            print('X%d < %.3f Gini=%.3f' % ((index + 1), row[index], gini))
            
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
                
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

dataset = [[2.771244718,1.784783929,0],
 [1.728571309,1.169761413,0],
 [3.678319846,2.81281357,0],
 [3.961043357,2.61995032,0],
 [2.999208922,2.209014212,0],
 [7.497545867,3.162953546,1],
 [9.00220326,3.339047188,1],
 [7.444542326,0.476683375,1],
 [10.12493903,3.234550982,1],
 [6.642287351,3.319983761,1]]
split = get_split(dataset)
print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))


def to_terminal(group):
 outcomes = [row[-1] for row in group]
 return max(set(outcomes), key=outcomes.count)