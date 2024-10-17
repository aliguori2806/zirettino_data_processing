import numpy as np

def OpenFileZFEB(file, dataversion=0, nevt2read = None):

    # Define data type to read the binary files (different dataversions are available)
    dtCitirocDNIs = {0: np.dtype([('Timestamp','<u8'),\
                      ('TriggerID','<u8'),\
                      ('DAQ1_ID','<u4'),\
                      ('DAQ1_TriggerCounts','<u4'),\
                      ('DAQ1_Valid','<u4'),\
                      ('DAQ1_Flag','<u4'),\
                      ('DAQ1_body','<u4',(128,)),\
                      ('DAQ1_Lost','<u8'),\
                      ('DAQ1_Validated','<u8'),\
                      ('DAQ2_ID','<u4'),\
                      ('DAQ2_TriggerCounts','<u4'),\
                      ('DAQ2_Valid','<u4'),\
                      ('DAQ2_Flag','<u4'),\
                      ('DAQ2_body','<u4',(128,)),\
                      ('DAQ2_Lost','<u8'),\
                      ('DAQ2_Validated','<u8')              
                      ]),
                    1: np.dtype([('Timestamp','<u8'),\
                      ('T1','<u4'),\
                      ('T0','<u4'),\
                      ('DAQ1_ID','<u4'),\
                      ('DAQ1_TriggerCounts','<u4'),\
                      ('DAQ1_Valid','<u4'),\
                      ('DAQ1_Flag','<u4'),\
                      ('DAQ1_body','<u4',(128,)),\
                      ('DAQ1_Lost','<u8'),\
                      ('DAQ1_Validated','<u8'),\
                      ('DAQ2_ID','<u4'),\
                      ('DAQ2_TriggerCounts','<u4'),\
                      ('DAQ2_Valid','<u4'),\
                      ('DAQ2_Flag','<u4'),\
                      ('DAQ2_body','<u4',(128,)),\
                      ('DAQ2_Lost','<u8'),\
                      ('DAQ2_Validated','<u8')              
                      ])
                  }
    dtCitirocDNI = dtCitirocDNIs[dataversion]
    
    # Read binary file into data
    if nevt2read != None:
        data = np.fromfile(file,dtype=dtCitirocDNI,offset=0,count=nevt2read)
    else:
        data = np.fromfile(file,dtype=dtCitirocDNI,offset=0)

    # Define time unit to convert timestamp data
    timestp_unit = 10e-3 #10ms

    # Reconstruct event dictionary from binary data
    if dataversion==0:
        events = {"Timestamp": (((data["Timestamp"] & 0xFFFFFFFF) << 32) + (data["Timestamp"] >> 32))*timestp_unit, #s
              "TriggerID": data["TriggerID"],
              "DAQ1_ID": data["DAQ1_ID"],
              "DAQ1_TriggerCounts": data["DAQ1_TriggerCounts"],
              "DAQ1_Valid": data["DAQ1_Valid"],
              "DAQ1_Flag": data["DAQ1_Flag"],      
                
              "DAQ1_HG": (data['DAQ1_body']>>0 & 0x3fff),
              "DAQ1_LG": (data['DAQ1_body']>>14 & 0x3fff),
              "DAQ1_Hit": (data['DAQ1_body']>>28 & 0x1),
              "DAQ1_Lost": data["DAQ1_Lost"],
              "DAQ1_Validated": data["DAQ1_Validated"],
    
              "DAQ2_ID": data["DAQ2_ID"],
              "DAQ2_TriggerCounts": data["DAQ2_TriggerCounts"],
              "DAQ2_Valid": data["DAQ2_Valid"],
              "DAQ2_Flag": data["DAQ2_Flag"],        
              "DAQ2_HG": (data['DAQ2_body']>>0 & 0x3fff),
              "DAQ2_LG": (data['DAQ2_body']>>14 & 0x3fff),
              "DAQ2_Hit": (data['DAQ2_body']>>28 & 0x1),
              "DAQ2_Lost": data["DAQ2_Lost"],
              "DAQ2_Validated": data["DAQ2_Validated"],
              }
    elif dataversion==1:
        events = {"Timestamp": (((data["Timestamp"] & 0xFFFFFFFF) << 32) + (data["Timestamp"] >> 32))*timestp_unit, #s
          "TriggerTag": data["T0"],
          "SpillID": data["T1"],
                  
          "DAQ1_ID": data["DAQ1_ID"],
          "DAQ1_TriggerCounts": data["DAQ1_TriggerCounts"],
          "DAQ1_Valid": data["DAQ1_Valid"],
          "DAQ1_Flag": data["DAQ1_Flag"],      
          "DAQ1_HG": (data['DAQ1_body']>>0 & 0x3fff),
          "DAQ1_LG": (data['DAQ1_body']>>14 & 0x3fff),
          "DAQ1_Hit": (data['DAQ1_body']>>28 & 0x1),
          "DAQ1_Lost": data["DAQ1_Lost"],
          "DAQ1_Validated": data["DAQ1_Validated"],
                  
          "DAQ2_ID": data["DAQ2_ID"],
          "DAQ2_TriggerCounts": data["DAQ2_TriggerCounts"],
          "DAQ2_Valid": data["DAQ2_Valid"],
          "DAQ2_Flag": data["DAQ2_Flag"],        
          "DAQ2_HG": (data['DAQ2_body']>>0 & 0x3fff),
          "DAQ2_LG": (data['DAQ2_body']>>14 & 0x3fff),
          "DAQ2_Hit": (data['DAQ2_body']>>28 & 0x1),
          "DAQ2_Lost": data["DAQ2_Lost"],
          "DAQ2_Validated": data["DAQ2_Validated"],
          }
    return events