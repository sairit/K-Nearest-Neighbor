"""
@Author: Sai Yadavalli
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class KNearestNeighbor:
    def __init__ (self, k, featureB):
        """
        Initializes the K-Nearest-Neighbor model.
        
        Args:
            k (int): The number of neighbors to consider.
            featureB (str): The column to be classified.
            path (str): The path to the data files.
        """

        self.data = None
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.featureB = featureB
        self.k = k

    def load_data(self):
        """
        Loads data from a CSV file specified by the user.

        Returns:
            pd.DataFrame: Loaded dataset.
        """

        # Ask user for file name
        file_path = input("Enter the file path for your data: ")
        
        # Read data and print column names
        df = pd.read_csv(file_path)
        print("The data headers are: ", list(df))

        self.data = df
        return self.data
    
    def split_data(self):
        """
        Splits the dataset into five folds for cross-validation and saves them as CSV files.
        """

        if self.data is None:
            self.load_data()

        if not os.path.exists(f"{self.path}/train"):
            os.makedirs(f"{self.path}/train")

        df = self.data

        # Split data into 5 folds
        f1 = df.sample(frac=0.2)
        df2 = df.drop(f1.index)
        f2 = df2.sample(frac=0.25)
        df3 = df2.drop(f2.index)
        f3 = df3.sample(frac=(1/3))
        df4 = df3.drop(f3.index)
        f4 = df4.sample(frac=0.5)
        f5 = df4.drop(f4.index)

        # Concatenate the folds to create training and testing sets
        train1 = pd.concat([f2,f3,f4,f5])
        train2 = pd.concat([f1,f3,f4,f5])
        train3 = pd.concat([f1,f2,f4,f5])
        train4 = pd.concat([f1,f2,f3,f5])
        train5 = pd.concat([f1,f2,f3,f4])

        # Save test and validation sets as separate files
        train1.to_csv(f"{self.path}/train/train1.csv", index = False)
        f1.to_csv(f"{self.path}/train/validation1.csv", index = False)
        train2.to_csv(f"{self.path}/train/train2.csv", index = False)
        f2.to_csv(f"{self.path}/train/validation2.csv", index = False)
        train3.to_csv(f"{self.path}/train/train3.csv", index = False)
        f3.to_csv(f"{self.path}/train/validation3.csv", index = False)
        train4.to_csv(f"{self.path}/train/train4.csv", index = False)
        f4.to_csv(f"{self.path}/train/validation4.csv", index = False)
        train5.to_csv(f"{self.path}/train/train5.csv", index = False)
        f5.to_csv(f"{self.path}/train/validation5.csv", index = False)

    def makeA(self, df,col):
        """
        Creates a matrix A from a DataFrame by dropping a specified column.

        Args:
            df (pd.DataFrame): The DataFrame to be converted.
            col (str): The column to be dropped.

        Returns:
            np.ndarray: The matrix A.
        """

        tdf = df.drop(col,axis=1)
        A = tdf.to_numpy()
        return A
    
    def distance(self, a,b,n):
        """
        Calculates the Euclidean distance between two points.

        Args:
            a (np.ndarray): The first point.
            b (np.ndarray): The second point.
            n (int): The number of features.
        
        Returns:
            float: The Euclidean distance between the two points.
        """

        ones = np.ones((n,1))
        C = (a-b)*(a-b)
        temp = np.dot(C, ones)
        dist = temp**.5
        return dist
    
    def count_neighbors(self, k, sdf):
        """
        Counts the number of neighbors of each class in a sorted
        DataFrame and returns the class with the most neighbors.

        Args:
            k (int): The number of neighbors to consider.
            sdf (pd.DataFrame): The sorted DataFrame.
        
        Returns:
            str: The classification with the most neighbors.
        """

        counts = {}
        for i in range(k):
            for classification in sdf[self.featureB].unique():
                if classification not in counts:
                    counts[classification] = 0
                if sdf.loc[i].at[self.featureB] == classification:
                    counts[classification] += 1
            
        return max(counts, key=counts.get)
        
    def standardize(self, df, label):
        """
        Standardizes the features of a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be standardized.
            label (str): The column to be classified.

        Returns:
            pd.DataFrame: The standardized DataFrame.
        """

        df_copy = df.copy(deep=True)
        for feature in df_copy.columns:
            if feature != label:
                mean = df_copy[feature].mean()
                std = df_copy[feature].std(ddof=0)
                df_copy[feature] = (df_copy[feature] - mean) / std
        
        return df_copy
    
    def testing(self, positive_class, test, train, standard=False):
        """
        Tests the K-Nearest-Neighbor model using a test dataset.

        Args:
            test (str): Path to the test dataset.
            positive_class (str): The positive class to be used.

        Returns:
            list: A list containing the accuracy, precision, recall, and F1 score (in that order)
        """

        df = pd.read_csv(train) 
        testdf = pd.read_csv(test)
        
        if standard:
            df = self.standardize(df, self.featureB)
            testdf = self.standardize(testdf, self.featureB)
        
        train_features = []
        for column in df.columns:
            if column != self.featureB:
                train_features.append(column)

        n = len(df.columns)-1
        testm = len(testdf)
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(testm):
            values = testdf.loc[i, train_features].values 
            b= np.array([values])
            dropCol = self.featureB
            A = self.makeA(df, dropCol)
            dist_array = self.distance(A,b,n)
            tdf = df.copy(deep=True)
            tdf.insert(0, "dist", dist_array.flatten())
            sortedTdf= tdf.sort_values("dist")
            sortedTdf.reset_index(inplace=True, drop = True)
            sortedTdf
            ans = self.count_neighbors(self.k, sortedTdf)
            if ans == testdf.loc[i].at[self.featureB] and ans == positive_class:
                TP = TP+1
            elif ans == testdf.loc[i].at[self.featureB]:
                TN = TN +1
            elif ans == positive_class:
                FP = FP+1
            else: 
                FN = FN+1 

        print("TP:",TP)
        print("FP:",FP)
        print("TN:",TN)
        print("FN:",FN)
        Accuracy = (TP+TN)/testm
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1 = 1/((1/Precision)+1/Recall)
        Errors = FN + FP

        print("Accuracy:", Accuracy)
        print("Precision:", Precision)
        print("Recall:", Recall)
        print("F1:", F1)

        return [Accuracy, Precision, Recall, F1, Errors]
    
    def training(self, standard=False):
        """
        Trains the K-Nearest-Neighbor model using the training data.
        """

        print()
        print("Analyzing Optimal K Value")
        print()
        scores = []
        ks = []
        for i in range(1, self.k+1):
            self.k = i
            train_scores = []
            ks.append(i)
            print()
            print("Testing k=", i)
            print()
            for j in range(5):
                print(f"Fold {j}")
                train = f"{self.path}/train/train{j+1}.csv"
                validation = f"{self.path}/train/validation{j+1}.csv"
                train_scores.append(self.testing("M", validation, train, standard)[3])
            scores.append(np.mean(train_scores))

        plt.figure()
        plt.plot(ks, scores)
        plt.xlabel("K")
        plt.ylabel("F1 Score")
        plt.title("F1 Score vs. K")
        plt.savefig(f"{self.path}/figures/k_vs_f1.png")
        plt.show()


    def error_table(self, standard=False):
        """
        Creates a table of errors for different values of k.

        Args:
            ks (list): A list of k values to test.

        Returns:
            pd.DataFrame: A DataFrame containing the errors for each k value.
        """

        print()
        print("Creating Error Table")
        print()

        ks = range(1, self.k+1)
        table = []
        titles = []
        for i in range(1, 6):
            titles.append(f"Test {i} Errors")
        titles.append("TOTALS")
        table.append(titles)    
        
        for k in ks:
            self.k = k
            errors = []
            print()
            print(f"Testing k={k}")
            print()
            for i in range(1, 6):
                print(f"Train set on fold {i}")
                train = f"{self.path}/train/train{i}.csv"
                validation = f"{self.path}/train/validation{i}.csv"
                errors.append(self.testing("M", validation, train, standard)[4])
            errors.append(sum(errors))
            table.append(errors)

        table = np.array(table).T
        table = pd.DataFrame(table)
        columns = []
        columns.append("k")
        for k in ks:
            columns.append(f"{k}")
        table.columns = columns
        
        if not os.path.exists(f"{self.path}/figures"):
            os.makedirs("figures")
        
        table.to_csv(f"{self.path}/figures/error_table.csv", index=False)
        return table
        

            
# MAIN
# Testing the K-Nearest-Neighbor model
model = KNearestNeighbor(5, "Diagnosis")

# model.split_data()
train = input("Enter the file path to the training data: ")
validation = input("Enter the file path to the test data: ")
model.testing("M", validation, train, standard=True)


