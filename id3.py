#!/usr/bin/env python3

# PROGRAMMER: Ruth Dohrmann
# PROGRAM: id3.py
# DUE DATE: 11/15/23
# COURSE: Intro Artificial Intelligence (CSCI-4350-001)
# INSTRUCTOR: Dr. Joshua Phillips
# 
# Description: This program creates an ID3 decision tree to classify a set of input
# training data and then reports the classification performance on a separate set of 
# input validation data

import sys
import numpy as np

class node():
    # node includes the split value, attribute column, whether or not the node is a leaf,
    # the child with attribute values less than the split value, the child with attribute
    # values greater than or equal to the split value, and the end classification (this
    # is only applicable if the node is a leaf node).
    def __init__(self, split_val, column, leaf, childLT=None, childGE=None, classification=""):
        self.split_val = split_val
        self.column = column
        self.leaf = leaf
        self.childLT = childLT
        self.childGE = childGE
        self.classification = classification

    # Getter functions
    def getSplitVal(self):
        return self.split_val
    def getColumn(self):
        return self.column
    def getLeaf(self):
        return self.leaf
    def getChildLT(self):
        return self.childLT
    def getChildGE(self):
        return self.childGE
    def getClassification(self):
        return self.classification

    def __str__(self):
        # return string of board
        return f'Node: {self.split_val=}, {self.leaf=}, {self.column=}, {self.classification=}'

def main():
    # get training data
    fileName = sys.argv[1]
    data = np.loadtxt(fileName)

    # build the decision tree
    decision_tree = decision_tree_learning(data)

    # get validation data
    newFileName = sys.argv[2]
    val_data = np.loadtxt(newFileName)
    num_correct = 0

    # check if the validation data is only one row long
    if len(val_data.shape) < 2:
        num_correct += test_tree(val_data, decision_tree)
    else:
        # find the number of correctly classified objects
        for i in range(val_data.shape[0]):
            num_correct += test_tree(val_data[i], decision_tree)
    print(f"{num_correct}")


# Build the decision tree
def decision_tree_learning(data):

    # check for edge case
    if len(data.shape) < 2:
        return node(None, None, True, classification=data[-1])

    # Sort all columns (just retain sorted indices)
    indices = np.argsort(data,axis=0)

    # find initial entropy 
    uniq = np.unique(data[:,data.shape[1]-1], return_counts=True, axis=0)
    initial_prob = (uniq[1]/(uniq[1].sum()))
    initial_entropy = (-(initial_prob)*np.log2(initial_prob))
    initial_entropy = initial_entropy.sum()

    # if there is no initial uncertainty, return leaf node
    if initial_entropy == 0:
        return node(np.nan, np.nan, True, classification=uniq[0][0])

    max_info_gain = 0
    best_split_val = 0
    split_column = 0
    best_split_subsetLT = best_split_subsetGE = data


    # Proceed for each column
    for x in range(data.shape[1]-1):
        previous = data[indices[0,x],x]
        current = data[indices[0,x],x]
        # Go through column row-by-row
        for y in indices[:,x]:
            # update current attribute value
            current = data[y,x]
            # check if there is a potential split in the data
            if previous < current:
                # find split point
                split_val = [(data[indices[:,x],x]).tolist().index(current)]
                # split the data
                split = np.split(data[indices[:,x],:], split_val, axis=0)
                # find entropy
                new_entropy = entropy_part(data[:,x], split[0]) + entropy_part(data[:,x], split[1])
                # calculate information gain
                new_info_gain = initial_entropy - new_entropy
                # if new information gain is greater, update split point
                if new_info_gain > max_info_gain:
                    max_info_gain = new_info_gain
                    best_split_val = (current + previous) / 2
                    split_column = x
                    best_split_subsetLT = split[0]
                    best_split_subsetGE = split[1]
            previous = data[y, x]
     
    # if a split was not possible or a split did not end in a greater information gain, return
    # a leave node with the maximum number of classifications as its classification value
    if max_info_gain == 0:
        return node(None, None, True, classification=uniq[0][uniq[1].tolist().index((uniq[1].max()))])
    # otherwise, split at the best point and recursively call the function decision_tree_learning
    # to find the children nodes
    else:
        return node(best_split_val, split_column, False, decision_tree_learning(best_split_subsetLT), decision_tree_learning(best_split_subsetGE))


# Find the entropy for one section of the data
def entropy_part(data, subset):
    # the probability of randomly selecting this portion of the data
    sub_prob = subset.shape[0] / data.shape[0]

    # the number of unique classifications in the data as well as the number of occurrences of each
    uniq = np.unique(subset[:,subset.shape[1]-1], return_counts=True, axis=0)
    uniq = uniq[1]
    # the number of rows
    initial_prob = uniq.sum()

    # the number of unique classifications in the data
    uniq_indiv = len(uniq)
    return_sum = 0

    # calculate and sum individual amounts of entropy
    for x in range(uniq_indiv):
        # the probability that the current classification will be selected
        # (only looking at the subset)
        curr_prob = uniq[x] / initial_prob
        info_content = -np.log2(curr_prob)
        return_sum += (curr_prob * info_content)

    return (sub_prob * return_sum)

# Return one if the object was classified correctly, otherwise return 0
def test_tree(object, tree):
    # if the node is not a leaf, select the correct child and move down the tree
    if not tree.leaf:
        column = tree.getColumn()
        value = tree.getSplitVal()
        if object[column] < value:
            return test_tree(object, tree.getChildLT())
        else:
            return test_tree(object, tree.getChildGE())
    # if the node is a leaf, check if the node was classified correctly
    else:
        classification = tree.getClassification()
        if classification == object[-1]:
            return 1
        else:
            return 0


if __name__ == "__main__":
    main()