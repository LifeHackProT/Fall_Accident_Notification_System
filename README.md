# Fall_Accident_Notification_System
using IMU sensor

[Development Sequence]
1. Data Collect - walking and dumping, running, lying
2. Data Preprocessing using MatLab 
  - Data labeling, Feature emphasize 6 to 40
  - Fix the window size : Extract specific part
  - And save all the thing
3. Make a python code
  - GA : Genetic Algorithm -> We have 40 Feature and than system has a overload so resolve that large data and Increase accuracy
  - Run 10 time : 
    The number of 100 cases is output. 
    The frequency of each feature is output and forward selection is performed from the most frequent to the smallest number until optimal accuracy is achieved.
    We getting a 15
4. Decision Tree
  - Reason for use: Simple mechanism, high accuracy and fast speed
  - Learning data received from MatLab
  - In real time, data received from Arduino is used as test data to determine whether it is a fall or ADL based on what has been learned

=> Transmit information by connecting with emergency alarm application
