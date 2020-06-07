"""Get request to TickData server, to fetch data using API.
    Work in progress, request response returns access denied somehow...
    Please contact author for access credentials

    Author :
        Alexandre Bremard
    Contributors :
        -
    Version Control :
        0.1 - 07/06/2020 : get
"""

# ----------------------------------------------------------------- External Imports
import requests
import os.path
import urllib.request as url
import urllib.error as urlerror


# ----------------------------------------------------------------- Body
def get(symbollist_path = "../Data/Output/Production/SymbolList.csv",
        save_path = "../Data/Output/Production/"):
    """API get request

    Args:
        symbollist_path (str, optional): Path to CSV file containing symbols. Defaults to "../Data/Output/Production/SymbolList.csv".
        save_path (str, optional): Path to folder where CSV files will be saved. Defaults to "../Data/Output/Production/".
    """        

    ## API credentials and Parameters
    api_user = "*****"
    api_pwd = "*****"
    StartDate = '01/04/2020'
    EndDate = '01/08/2020'

    ## Read Symbol List File
    with open(symbollist_path) as f2:
        symbollist = f2.readlines()
    symbollist = [x.strip() for x in symbollist]
    f2.close()

    ## Request Data
    '''
    The url was built using Tick Data's RequestBuilder: https://sandbox-tickapi.tickdata.com/
    Symbol, Start Data, and End Date have been parameterized.
    One minute bar data has been specified in the url.
    '''

    for symbol in symbollist:
        print("Downloading: ",symbol)

        ## Build the url
        urlp = 'https://tickapi.tickdata.com//stream?COLLECTION_TYPE=COLLECTION_TYPE.US_TED&EXTRACT_TYPE=COLLECTOR_SUBTYPE_US_TICK_TIMEBAR&DAYS_BACK=5&OUTPUT_NAMING_CONVENTION=0&DATE_FORMAT=MM/dd/yyyy&TIME_FORMAT=HH:mm&SELECTED_FIELDS=DATE_AND_TIME_FIELD|DATE_FIELD|TICK_TIME_FIELD|OPEN_PRICE_FIELD|HIGH_PRICE_FIELD|LOW_PRICE_FIELD|CLOSE_PRICE_FIELD|VOLUME_FIELD|UP_TICKS_FIELD|DOWN_TICKS_FIELD&REQUESTED_DATA=16841|AMZN&TIME_UNIT=MINUTES&GRANULARITY=5&EMPTY_INTERVAL_SETTING=0&EXTEND_LAST_INTERVAL=TRUE&NOTIFICATION_EMAIL=bremard.alexandre@gmail.com&INCLUDE_HEADER_ROW=TRUE&OUTPUT_TIME_ZONE=Exchange&PRICE_DECIMALS=8&COMPRESSION_TYPE=NONE'

        # create an OpenerDirector with support for Basic HTTP Authentication
        auth_handler = url.HTTPBasicAuthHandler()

        # adding authentication details the handler
        auth_handler.add_password(realm='TW Server',
                                uri=urlp,
                                user=api_user,
                                passwd=api_pwd)

        # building an opener auth handler
        opener = url.build_opener(auth_handler)
        url.install_opener(opener)

        # try block
        try:
            # opening the url
            f = url.urlopen(urlp)
            # opening the file for each symbol in the write binary mode
            with open(symbol + ".csv", "wb") as file:
                # writing the data from the URL to the file
                file.write(f.read())
        # exception handling
        except urlerror.HTTPError as e:
            # checking the HTTPError code
            if hasattr(e, 'code'):
                # checking for any other error apart from URL error
                if e.code != 401:
                    # printing the error
                    print('I have an error')
                    print(e.code)
            else:
                # printing the urlerror
                print(e.headers)


# ----------------------------------------------------------------- Test
def test():
    """This function is internal to GET.py, it is meant for debugging but also serves as unit test
    """
    print("----- TEST FOR INDICATORS.PY -----")
    print("----- GET ------")
    get()
    print("----------------------------------")