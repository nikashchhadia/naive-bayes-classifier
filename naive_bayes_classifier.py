import numpy as np
import pandas as pd

def load(filename, include_demographics=True):
    df = pd.read_csv(f"{filename}")
    if not include_demographics:
        df = df.drop(columns=["Demographic"])
    
    return df

def get_p_x_given_y(x_column, y_column, df):
    # returns [P(X = 1 | Y = 0), P(X = 1 | Y = 1)]

    y1 = sum(df[y_column])  # the number of rows where Y = 1
    y0 = len(df[y_column]) - y1  # the number of rows where Y = 0
    x0 = 0  # the number of rows where X = 1 | Y = 0
    x1 = 0  # the number of rows where X = 1 | Y = 1
    n = df.shape[0]  # number of rows
    for i in range(n):
        if df[y_column][i] == 0 and df[x_column][i] == 1:
            x0 += 1
        elif df[y_column][i] == 1 and df[x_column][i] == 1:
            x1 += 1

    # Laplace estimations
    p_0 = (x0 + 1) / (y0 + 2)
    p_1 = (x1 + 1) / (y1 + 2)
    
    return [p_0, p_1]

def get_all_p_x_given_y(y_column, df):
    # stores P(X_i=1 | Y=y) in all_p_x_given_y[i][y]

    all_p_x_given_y = np.zeros((df.shape[1] - 1, 2))  # array for each P(X_i=1 | Y=y)
    columns = df.columns
    m = len(columns) - 1  # just including the features, so need to remove Label
    for i in range(m):
        p_x_given_y = get_p_x_given_y(columns[i], y_column, df)
        all_p_x_given_y[i][0] = p_x_given_y[0]
        all_p_x_given_y[i][1] = p_x_given_y[1]

    return all_p_x_given_y

def get_p_y(y_column, df):
    # returns P(Y = 1)
    return df[y_column].mean()

def joint_prob(xs, y, all_p_x_given_y, p_y):
    # returns the joint probability of a single row and y
    # this is equal to P(X, Y) = P(Y) * P(X_1 | Y) * P(X_2 | Y) * ... * P(X_n | Y)

    prob = p_y
    for i, x in enumerate(xs):
        if x == 1:
            prob *= all_p_x_given_y[i][y]  # P(X_i = 1 | Y)
        elif x == 0:
            prob *= (1 - all_p_x_given_y[i][y])  # P(X_i = 0 | Y)

    return prob

def compute_accuracy(all_p_x_given_y, p_y, df):
    # split the test set into X and y. The predictions should not be able to refer to the test y's.
    X_test = df.drop(columns="Label")
    y_test = df["Label"]

    num_correct = 0
    total = len(y_test)

    for i, xs in X_test.iterrows():
        # we can make our prediction by taking the argmax (wrt y) of the joint prob
        p0 = joint_prob(xs, 0, all_p_x_given_y, 1 - p_y)
        p1 = joint_prob(xs, 1, all_p_x_given_y, p_y)
        if (p0 >= p1 and y_test[i] == 0) or (p1 > p0 and y_test[i] == 1):
            num_correct += 1  # if our prediction is correct, increment num_correct

    return num_correct / total  # return the computed accuracy

def main():
    # load the training set
    df_train = load("heart-train.csv")

    # compute model parameters (i.e. P(Y), P(X_i|Y))
    all_p_x_given_y = get_all_p_x_given_y("Label", df_train)
    p_y = get_p_y("Label", df_train)

    # load the test set
    df_test = load("heart-test.csv")

    print(f"Training accuracy: {compute_accuracy(all_p_x_given_y, p_y, df_train)}")
    print(f"Test accuracy: {compute_accuracy(all_p_x_given_y, p_y, df_test)}")

if __name__ == "__main__":
    main()