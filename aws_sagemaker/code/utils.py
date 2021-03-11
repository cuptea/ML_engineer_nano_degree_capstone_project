import numpy as np

def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan
    
def isrc_to_country(isrc):
    if type(isrc) == str:
        return isrc[:2]
        
    else:
        return np.nan
    
def bd_to_age(bd):
    if int(bd)>80 or int(bd)<=6:
        return int(bd)
    else:
        return 0