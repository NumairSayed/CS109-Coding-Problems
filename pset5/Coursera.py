import pandas as pd
import numpy as np
from tqdm import tqdm

def problem_one_and_two():
    # Load data from CSV with custom column names
    df = pd.read_csv('/home/numair/Desktop/Codes/Python/Stanford Psets/Pset4_ChrisPiechMoule/pset5/learningOutcomes.csv', names=['ID', 'Activity', 'Score'])

    # Filter rows based on 'Activity' column
    dfA = df[df['Activity'] == 'activity1']
    dfB = df[df['Activity'] == 'activity2']

    # Calculate mean score for each subset
    meanA = dfA['Score'].mean()
    meanB = dfB['Score'].mean()

    print("Mean Score for activity1:", meanA)
    print("Mean Score for activity2:", meanB)
    observed_mean_diff = meanB - meanA
    print(f"MeanB - MeanA = {observed_mean_diff}")
    print(len(dfA), len(dfB))

    # Bootstrap sampling and probability calculation
    sample_size = 100000
    list_diff = []
    for _ in tqdm(range(sample_size)):
        sampleA = dfA.sample(n=len(dfA), replace=True)
        sampleB = dfB.sample(n=len(dfB), replace=True)
        sampled_meanA = sampleA['Score'].mean()
        sampled_meanB = sampleB['Score'].mean()
        sampled_mean_diff = sampled_meanB - sampled_meanA
        list_diff.append(sampled_mean_diff)

    freq = sum(1 for diff in list_diff if diff >= observed_mean_diff) / sample_size
    print(f"The probability of observing same or greater mean diff = {freq}")

def calculate_mean_diff(df, activity1, activity2, sample_size=100000):
    list_diff = []
    for _ in tqdm(range(sample_size)):
        sampleA = df[df['Activity'] == activity1].sample(n=len(df[df['Activity'] == activity1]), replace=True)
        sampleB = df[df['Activity'] == activity2].sample(n=len(df[df['Activity'] == activity2]), replace=True)
        sampled_meanA = sampleA['Score'].mean()
        sampled_meanB = sampleB['Score'].mean()
        sampled_mean_diff = sampled_meanB - sampled_meanA
        list_diff.append(sampled_mean_diff)
    return list_diff

def problem_three():
    # Load data
    dfA = pd.read_csv('/home/numair/Desktop/Codes/Python/Stanford Psets/Pset4_ChrisPiechMoule/pset5/learningOutcomes.csv', names=['ID', 'Activity', 'Score'])
    dfB = pd.read_csv('/home/numair/Desktop/Codes/Python/Stanford Psets/Pset4_ChrisPiechMoule/pset5/background.csv', names=['ID', 'Experience'])

    # Merge data
    dfA = pd.merge(dfA, dfB, on='ID')

    # Filter by experience level
    dfLess = dfA[dfA['Experience'] == 'less']
    dfAvg = dfA[dfA['Experience'] == 'average']
    dfMore = dfA[dfA['Experience'] == 'more']

    # Subset activities
    df1Less = dfLess[dfLess['Activity'] == 'activity1']
    df2Less = dfLess[dfLess['Activity'] == 'activity2']

    df1Avg = dfAvg[dfAvg['Activity'] == 'activity1']
    df2Avg = dfAvg[dfAvg['Activity'] == 'activity2']

    df1More = dfMore[dfMore['Activity'] == 'activity1']
    df2More = dfMore[dfMore['Activity'] == 'activity2']

    # Calculate observed mean differences
    mean_diff_Less = df2Less['Score'].mean() - df1Less['Score'].mean()
    mean_diff_Avg = df2Avg['Score'].mean() - df1Avg['Score'].mean()
    mean_diff_More = df2More['Score'].mean() - df1More['Score'].mean()

    # Bootstrap sampling and probability calculation
    freqLess = sum(1 for diff in calculate_mean_diff(dfLess, 'activity1', 'activity2') if diff >= mean_diff_Less) / 100000
    print("\n")
    print("***************************************************************************************************************")
    print("\n")
    
    freqAvg = sum(1 for diff in calculate_mean_diff(dfAvg, 'activity1', 'activity2') if diff >= mean_diff_Avg) / 100000

    print("\n")
    print("***************************************************************************************************************")
    print("\n")

    freqMore = sum(1 for diff in calculate_mean_diff(dfMore, 'activity1', 'activity2') if diff >= mean_diff_More) / 100000
    
    print("\n")
    print("***************************************************************************************************************")
    print("\n")

    print(f"Probability for less experience: {freqLess}")
    print(f"Probability for average experience: {freqAvg}")
    print(f"Probability for more experience: {freqMore}")


def main():
    problem_one_and_two()
    problem_three()


if __name__ == '__main__':
    main()
