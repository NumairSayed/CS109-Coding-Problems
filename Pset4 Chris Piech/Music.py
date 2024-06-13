import numpy as np
import pandas as pd


def prob(df ,music_type , inputrandomvariable )->float: #prob(pandas DataFrame, musictype, randomvariable)
    prob=0
    for index,rows in df.iterrows():
        if (rows[music_type] == inputrandomvariable):
            prob+=1
    return prob/len(df)   


def expectation(df,music_type)->float:
    freq=0
    for index,rows in df.iterrows():
        freq+= rows[music_type]
    return freq/len((df))


def cond_prob(df, music_type_to_find ,x, music_type_given,y )->float: #conditional prob -> p(music_type_to_find = x / music_type_given = y)
    numerator   = 0
    denominator = 0
    for index,rows in df.iterrows():
        if (rows[music_type_to_find]==x and rows[music_type_given]==y):
            numerator+=1
        if(rows[music_type_given]==y):
            denominator+=1

    return numerator/denominator


def corr(df, param1, param2,normalized):
    numerator = 0
    covariance = 0
    param1mean = df[param1].mean()
    param2mean = df[param2].mean()
    for index,rows in df.iterrows():
        numerator+= ((rows[param1] - param1mean) * (rows[param2] - param2mean))
    covariance = numerator / (len(df)-1)
    if normalized:
        return (covariance / (df[param1].std() * df[param2].std()))  ## Calculated std deviations on the fly
    else:
        return covariance

def main():
    df = pd.read_csv('music.csv')

    #print(df.iloc[45]['Pop'])
    print(f"Probability of Folk music with perfect rating is:{prob(df,'Folk',5)}")
    print(f"Average Ratings of Folk music is: {expectation(df,'Folk')}")
    print(f"Probability Folk Music is rated 5 given that Musical is rated 5: {cond_prob(df,'Folk',5,'Musical',5)}")
    print(f" Correlation between Opera and Punk type music is: {corr(df,'Opera','Punk',False)}") # Keep Normalized True to obtain correlation [-1,1]



if __name__ == '__main__':
    main()
