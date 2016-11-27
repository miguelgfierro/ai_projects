country_dict = {'ALB':'Albania',
                'AND':'Andorra',
                'ARM':'Armenia',
                'AUT':'Austria',
                'AZE':'Azerbaijan',
                'BEL':'Belgium',
                'BIH':'Bosnia and Herzegovina',
                'BLR':'Belarus',
                'BUL':'Bulgaria',
                'CRO':'Croatia',
                'CYP':'Cyprus',
                'CZE':'Czech Republic',
                'DEN':'Denmark',
                #'ENG':'England',
                'ENG':'United Kingdom',
                'ESP':'Spain',
                'EST':'Estonia',
                'FIN':'Finland',
                'FRA':'France',
                'FRO':'Feroe Islands',
                'GEO':'Georgia',
                'GER':'Germany',
                'GIB':'Gibraltar',
                'GRE':'Greece',
                'HUN':'Hungary',
                'ITA':'Italy',
                'IRL':'Ireland',
                'ISL':'Iceland',
                'ISR':'Israel',
                'KAZ':'Kazakhstan',
                'LTU':'Lithuania',
                'LUX':'Luxembourg',
                'LVA':'Latvia',
                'MDA':'Moldova',
                'MKD':'Macedonia',
                'MLT':'Malta',
                'MNE':'Montenegro',
                'NED':'Netherlands',
                #'NIR':'Northern Ireland',
                'NIR':'United Kingdom',
                'NOR':'Norwey',
                'POL':'Poland',
                'POR':'Portugal',
                'ROU':'Romania',
                'RUS':'Russia',
                #'SCO':'Scotland',
                'SCO':'United Kingdom',
                'SMR':'San Marino',
                'SRB':'Serbia',
                'SUI':'Switzerland',
                'SVK':'Slovakia',
                'SVN':'Slovenia',
                'SWE':'Sweden',
                'TUR':'Turkey',
                'UKR':'Ukrania',
                #'WAL':'Wales',
                'WAL':'United Kingdom'}


def to_web_mercator(yLat, xLon):
    # Check if coordinate out of range for Latitude/Longitude
    if (abs(xLon) > 180) and (abs(yLat) > 90):  
        return
 
    semimajorAxis = 6378137.0  # WGS84 spheriod semimajor axis
    east = xLon * 0.017453292519943295
    north = yLat * 0.017453292519943295
 
    northing = 3189068.5 * math.log((1.0 + math.sin(north)) / (1.0 - math.sin(north)))
    easting = semimajorAxis * east
 
    return [easting, northing]

def aggregate_dataframe_coordinates(dataframe):
    df = pd.DataFrame(index=np.arange(0, n_complete_matches*3), columns=['Latitude','Longitude'])
    count = 0
    for ii in range(dataframe.shape[0]):
        if dataframe['home_stadium'].loc[ii]!= 'Unknown' and dataframe['visitor_stadium'].loc[ii]!= 'Unknown':
            df.loc[count] = [dataframe['home_latitude'].loc[ii], dataframe['home_longitude'].loc[ii]]
            df.loc[count+1] = [dataframe['visitor_latitude'].loc[ii], dataframe['visitor_longitude'].loc[ii]]
            df.loc[count+2] = [np.NaN, np.NaN]
            count += 3
    return df




