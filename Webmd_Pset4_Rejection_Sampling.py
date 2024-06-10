import numpy as np
import pandas as pd

# This is a tuple to ascertain ordering of boolean values of risks - disease - symptoms
columns = ['stress', 'exposure', 'cold', 'flu', 'x1', 'x2', 'x3', 'x4', 'x5']

#independent and unconditioned probability od stress
def probStress():
    return 0.5

#independent and unconditioned probability of exposure 
def probExposure():
    return 0.1

# Parents of Cold in Bayesian Network were Stress and Exposure, so Cold is conditioned on its parents
def probCold(s, e):
    if s == 0 and e == 0:
        return 0.01
    if s == 0 and e == 1:
        return 0.2
    if s == 1 and e == 0:
        return 0.3
    if s == 1 and e == 1:
        return 0.7
    raise IndexError("valid parameters are [0,1]")


# Parents of Flu in Bayesian Network were Stress and Exposure, so Flu is conditioned on its parents
def probFlu(s, e):
    if s == 0 and e == 0:
        return 0.01
    if s == 0 and e == 1:
        return 0.5
    if s == 1 and e == 0:
        return 0.1
    if s == 1 and e == 1:
        return 0.8
    raise IndexError("valid parameters are [0,1]")

_symptom_dict = {1: [0.03, 0.20, 0.80, 0.60],
                 2: [0.05, 0.70, 0.60, 0.80],
                 3: [0.03, 0.50, 0.50, 0.90],
                 4: [0.02, 0.30, 0.90, 0.50],
                 5: [0.01, 0.80, 0.80, 0.60]}

# Parents of each symptom Xi in Bayesian Network were Flu and Cold, so they are conditioned on their parents
def probSymptom(i, f, c):
    bin_val = 2 * f + c  # decimal conversion of binary number fc
    return _symptom_dict[i][bin_val]

#This helps a lot.
def bern(prob):
    return 1 if np.random.rand() < prob else 0

#Fn to infer probability
def inferProbFlu(ntrials=1000000) -> float:
    rows = [] #Empty list to represent status of any sample from population.

    for _ in range(ntrials):
        boolStress = bern(probStress())
        boolExp = bern(probExposure())
        boolCold = bern(probCold(boolStress, boolExp))
        boolFlu = bern(probFlu(boolStress, boolExp))
        boolX1 = probSymptom(1, boolFlu, boolCold)
        boolX2 = probSymptom(2, boolFlu, boolCold)
        boolX3 = probSymptom(3, boolFlu, boolCold)
        boolX4 = probSymptom(4, boolFlu, boolCold)
        boolX5 = probSymptom(5, boolFlu, boolCold)
        rows.append({
            'stress': boolStress,
            'exposure': boolExp,
            'cold': boolCold,
            'flu': boolFlu,
            'x1': boolX1,
            'x2': boolX2,
            'x3': boolX3,
            'x4': boolX4,
            'x5': boolX5
        })

    df = pd.DataFrame(rows) #Initialize dataframe with given schema of rows(rows is declared globally)
    df_filtered = df[(df['exposure'] == 1) & (df['x2'] > 0)] 
    """Filtered samples put in dataframe through list comprehension
       We call this rejection sampling by pulling out subset of samples consistent with the "given that"
       condition of our conditional probability query
    """

    freq = df_filtered['flu'].sum()
    return freq / len(df_filtered) if len(df_filtered) > 0 else 0
def main():
    print("Calling inferProbFlu:")
    print("\tReturn value was:", inferProbFlu())
    print("Done!")

if __name__=='__main__':
    main()
    