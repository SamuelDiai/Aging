"""
20160	Ever smoked
20162	Pack years adult smoking as proportion of life span exposed to smoking ?
20161	Pack years of smoking ?
20116	Smoking status
1239	Current tobacco smoking
1249	Past tobacco smoking
2644	Light smokers, at least 100 smokes in lifetime
3436	Age started smoking in current smokers ?
3446	Type of tobacco currently smoked
5959	Previously smoked cigarettes on most/all days
3456	Number of cigarettes currently smoked daily (current cigarette smokers)
6194	Age stopped smoking cigarettes (current cigar/pipe or previous cigarette smoker)
6183	Number of cigarettes previously smoked daily (current cigar/pipe smokers)
3466	Time from waking to first cigarette
3476	Difficulty not smoking for 1 day
3486	Ever tried to stop smoking
3496	Wants to stop smoking
3506	Smoking compared to 10 years previous
6158	Why reduced smoking
2867	Age started smoking in former smokers
2877	Type of tobacco previously smoked
2887	Number of cigarettes previously smoked daily
2897	Age stopped smoking
2907	Ever stopped smoking for 6+ months

6157	Why stopped smoking

2926	Number of unsuccessful stop-smoking attempts
2936	Likelihood of resuming smoking
1259	Smoking/smokers in household
1269	Exposure to tobacco smoke at home
1279	Exposure to tobacco smoke outside home
"""

def read_smoking_data(instances = [0, 1, 2, 3], **kwargs):

        dict_onehot = {
            '20116' : {-3 : 'Prefer not to answer', 0 : 'Never', 1 : 'Previous', 2 : 'Current'},
            '1239' : {1 : 'Yes, on most or all days', 2 : 'Only occasionally', 0 : 'No', -3 : 'Prefer not to answer'},
            '1249' : {1 : 'Smoked on most or all days', 2 : 'Smoked occasionally', 3 : 'Just tried once or twice',
                      4 : 'I have never smoked', -3 : 'Prefer not to answer'},
            '2644' : {1 : 'Yes', 0 : 'No', -1 : 'Do not know', -3 : 'Prefer not to answer'},
            '3446' : {1 : 'Manufactured cigarettes', 2 : 'Hand-rolled cigarettes', 3 : 'Cigars or pipes', -7 : 'None of the above',
                      -3 : 'Prefer not to answer'},
            '3486' : {1 : 'Yes, tried but was not able to stop or stopped for less than 6 months', 2 : 'Yes, tried and stopped for at least 6 months',
                      0 : 'No', -3 : 'Prefer not to answer'},
            '6157' : {1 : 'Illness or ill health', 2 : "Doctor's advice",3 : 'Health precaution', 4 : 'Financial reasons', -7 : 'None of the above',
                     -1 : 'Do not know', -3 : 'Prefer not to answer'},
            '2936' : {1 : 'Yes, definitely', 2 : 'Yes, probably', 3 : 'No, probably not', 4 : 'No, definitely not', -1 : 'Do not know', -3 : 'Prefer not to answer'}
                      }

        cols_numb_onehot = {'20116' : 1, '1239' : 1, '1249' : 1, '2644' : 1, '3446' : 1, '6157' : 4, '2936' : 1}
        cols_ordinal = ['20160', '3466', '3476', '3496', '3506', '2907', '1259']
        cols_continuous = ['3456', '2867', '2897', '2926', '1269', '1279']
        cont_fill_na = {'3456' : 0, '2867' : 90, '2887' : 0, '1269' : 0, '1279' : 0}
        cols_half_binary = ['2907']
