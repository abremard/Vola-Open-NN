# Vola-Open-NN
    Neural Network for supervised Vola Open WL Strategy

# Prerequisites
    Python 3.8.3
    Tensorflow 2.1
    Keras 2.3.1

# Installing
    -

# Setup for production server
    !!! IMPORTANT !!!

    This project needs TickData (either TickWrite or TickWriteWeb) to run, please contact the owner of the project to obtain access.

    Important setups include :
    1) install TickWrite client
    2) creating a TickWrite schedule daily on week days that runs 2 separate TickWrite jobs:
        - one for tick data fetch, refresh new data for network training
        - one for time data fetch, required for next day success forecast
    3) On Windows, create task that runs SCHEDULE.PY using Task Scheduler, please refer to this tutorial
        https://www.youtube.com/watch?v=n2Cr_YRQk7o @Cedric Yarish


# GPU config
    To run on GPU using annaconda :
        First run :
            conda create -n tf-gpu tensorflow-gpu
        Activate :
            conda activate tf-gpu

# Unit test
    python TEST.py

# Authors
    Alexandre Bremard - Initial work

# License
    Use of this library contains sensitive data. Users must not use, share, or store any of the data. 
    (copyright© coming soon XD) 