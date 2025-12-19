from ucimlrepo import fetch_ucirepo

# fetch dataset
communities_and_crime = fetch_ucirepo(id=183)

def load_cac():

    # data (as pandas dataframes)
    X = communities_and_crime.data.features
    y = communities_and_crime.data.targets

    # metadata
    print(communities_and_crime.metadata)

    # variable information
    print(communities_and_crime.variables)
    
    return X, y




