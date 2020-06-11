"""Python scheduled tasks that Windows executes daily
    Author :
        Alexandre Bremard
    Contributors :
        -
    Version Control :
        0.1 - 07/06/2020 : run job
"""

# f = open("guru99.txt","w+")
# for i in range(10):
#      f.write("This is line %d\r\n" % (i+1))
# f.close()

import PRODUCTION as production
production.job()