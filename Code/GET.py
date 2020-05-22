
import requests
import os.path
import urllib.request as url
import urllib.error as urlerror

# Location of symbol list to extract
symbollist_path = "Production/SymbolList.csv"

# Location to save output data
save_path = 'Production/'


## API credentials and Parameters
api_user = "philippe.bremard.idts@gmail.com"
api_pwd = "mogilno44385731"
StartDate = '01/04/2016'
EndDate = '01/08/2016'


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
    # urlp = 'https://tickapi.tickdata.com//stream?COLLECTION_TYPE=COLLECTION_TYPE.US_TED&EXTRACT_TYPE=COLLECTOR_SUBTYPE_US_TICK_TIMEBAR&DAYS_BACK=5&OUTPUT_NAMING_CONVENTION=0&DATE_FORMAT=MM/dd/yyyy&TIME_FORMAT=HH:mm&SELECTED_FIELDS=DATE_AND_TIME_FIELD|DATE_FIELD|TICK_TIME_FIELD|OPEN_PRICE_FIELD|HIGH_PRICE_FIELD|LOW_PRICE_FIELD|CLOSE_PRICE_FIELD|VOLUME_FIELD|UP_TICKS_FIELD|DOWN_TICKS_FIELD&REQUESTED_DATA=16841|AMZN&TIME_UNIT=MINUTES&GRANULARITY=5&EMPTY_INTERVAL_SETTING=0&EXTEND_LAST_INTERVAL=TRUE&NOTIFICATION_EMAIL=bremard.alexandre@gmail.com&INCLUDE_HEADER_ROW=TRUE&OUTPUT_TIME_ZONE=Exchange&PRICE_DECIMALS=8&COMPRESSION_TYPE=NONE'
    urlp = 'https://tickapi.tickdata.com//stream?COLLECTION_TYPE=COLLECTION_TYPE.US_TED' \
        #    '&EXTRACT_TYPE=COLLECTOR_SUBTYPE_US_TICK_TIMEBAR' \
        #    '&REQUESTED_DATA=' + symbol
    #        '&START_DATE=' + StartDate + \
    #        '&END_DATE=' + EndDate + \
    #        '&OUTPUT_NAMING_CONVENTION=0' \
    #        '&SESSIONS=MARKET' \
    #        '&GRANULARITY=1' \
    #        '&COMPRESSION_TYPE=NONE' \
    #        '&SELECTED_FIELDS=DATE_FIELD|TICK_TIME_FIELD|CLOSE_PRICE_FIELD|VOLUME_FIELD'
    # urlp = 'https://tickapi.tickdata.com//help?with=MARKET'


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

