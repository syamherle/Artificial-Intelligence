from __future__ import division
from collections import Counter
import math
import pickle

class DecisionTreeClassifier:

    def __init__(self,mails,top_words,model_file):
        self.threshold=3
        self.treeLength=1.0
        self.train(mails,top_words,model_file)

    def train(self, examples,top_words,model_file):
        messages = []
        words = top_words
        self.tree = {}
        for message in examples:
            messages.append(message)

        self.subtree(messages, words, 0)
        self.write_model(model_file)
        return self.tree

    def subtree(self, messages,words, depth):
        w = words
        m = messages
        d = depth
        if (len(m) <= self.threshold):
            c = 0
            for mess in m:
                if mess[-1] == 0:
                    c += 1
                else:
                    c -= 1
            if (c >= 0):
                cl = 0
            else:
                cl = 1
            self.tree[depth] = cl
            return
        temp = Counter()
        for message in m:
            temp[message[-1]] += 1

        if (float(temp[0]) / len(m) >= self.treeLength):
            self.tree[depth] = 0
            return
        elif (float(temp[1]) / len(m) >= self.treeLength):
            self.tree[depth] = 1
            return

        entropy=self.calculate_entropy(w,m)

        self.tree[depth] = entropy[1]
        leftTree = []
        rightTree = []
        if (not (entropy[1] == "#$%#")):
            for mess in m:
                if entropy[1] in mess:
                    leftTree.append(mess)
                else:
                    rightTree.append(mess)
            w.remove(entropy[1])
        self.subtree(leftTree, w, 2 * d + 2)
        self.subtree(rightTree, w, 2 * d + 1)

    def write_model(self,model_file):

        f = open(model_file, 'w')
        pickle.dump(self.tree, f)
        f.close()

    def calculate_entropy(self,w,m):
        entropy = (10000, "#$%#")
        classified = Counter()

        for word in w:
            for mess in m:
                if word in mess:
                    classified[word] += 1.0

        for word in classified:
            if (classified[word] == 1.0):
                w.remove(word)
        for word in classified:
            if (classified[word] > 1.0):
                y = float(classified[word] + 1) / float(len(m) + 2)
                n = float(float(len(m) + 2) - float(classified[word] + 1)) / float(len(m) + 2)
                e = float(-y * math.log(y) - n * math.log(n))
                if (e < entropy[0]):
                    entropy = (e, word)
        return entropy

def classify(sample,tree):
    return traverse(sample,tree, 0)

def traverse(message,tree, depth):

    if (tree[depth] == 0):
        return 0
    elif (tree[depth] == 0):
        return 1
    elif (tree[depth] in message):
        return traverse(message,tree, 2 * depth + 2)
    else:
        return traverse(message,tree, 2 * depth + 1)
