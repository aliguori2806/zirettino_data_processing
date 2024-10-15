import os
import numpy as np
import pandas as pd
import json
import uproot
import awkward as ak


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


def vPositionZFTK(strip_index, pitch=1., N_sipm_arrays=3, Nstrips1=32, sgn=1):
    """
    Function to compute position of strips considering the real array geometry: i.e. gaps
    
    Parameters:
    -----------
    -strip_index: (int) strip position in its own plane
    -pitch: (float) readout pitch in mm
    -N_sipm_arrays: (int) nr of Hamamatsu arrays on the pcb (e.g. zirettino 3, petrioc-ftk 1)
    -Nstrips1: (int) number of strips of the single array (e.g. OR4 32, OR1 128)
    -sgn: (int) according to position frame of reference axis wrt strip number
        +1 same direction
        -1 oppostie direction
        
    Returns:
    --------
    -geometrical position in mm
    """
    
    
    assert -0.1 <= strip_index <= (N_sipm_arrays*Nstrips1-1+.1)

    #Define geometrical parameters of the SiPM array
    int_gap = 0.220 #mm
    ext_gap = 0.380 #mm
    
    #Distance btw the origin and the center of the nearest SiPM array (if more than 1 exists)
    d = int_gap + ext_gap + pitch * Nstrips1 
    # print(f"Displacement unit (d) {d} .")
    
    #Index of the SiPM array where the strip with index strip_index belongs  
    sipm_array_index = int(strip_index / Nstrips1)  
    # print(f"sipm_array_index {sipm_array_index} .")
    
    sipm_array_center_position = sgn * d * 0.5 * (-N_sipm_arrays + 1 + 2 * sipm_array_index)
    # print(f"sipm_array_center_position {sipm_array_center_position} .")
    
    internal_strip_index = strip_index - sipm_array_index * Nstrips1
    # print(f"internal_strip_index {internal_strip_index} .")
    
    if internal_strip_index >= Nstrips1 / 2:
        internal_strip_position = sgn * (-pitch * 0.5 * (Nstrips1 - 1 - 2 * internal_strip_index) + 0.5 * int_gap)
    else:
        internal_strip_position = sgn * (-pitch * 0.5 * (Nstrips1 - 1 - 2 * internal_strip_index) - 0.5 * int_gap) 
    
    return sipm_array_center_position + internal_strip_position #mm


PositionZFTK = np.vectorize(vPositionZFTK, otypes=[np.float], cache=False)


class ZirettinoRun():

    def __init__(self):
        self.timestamp = None
        self.dataversion = None
        self.asic_info = None
        self.nfebs = 0
        self.data = None
        self.nevts = None
        self.configurators = None
        self.pedestals_dfs = None
        self.pedestal_subtracted_febs_daqs_gains = None
        self.gains_dfs = None
        self.charge_calibrated_febs_daqs_gains = None
        self.asics_data = None
        self.ftk_modules_data = None
        #self.ckeys = ["clu_ModuleID", "clu_PlaneID", "clu_SegmentID", "clu_ViewFlag", "clu_SideFlag", "clu_FirstStrip", "clu_Size", "clu_Charge_HG", "clu_ChargeStd_HG", "clu_Charge_LG", "clu_ChargeStd_LG", "clu_Position_HG", "clu_PositionError_HG", "clu_Position_LG", "clu_PositionError_LG"]
        self.eventkeys = ["Timestamp", "TriggerTag", "SpillID"]
        self.daq1keys = ["DAQ1_ID", "DAQ1_TriggerCounts", "DAQ1_Valid", "DAQ1_Flag", "DAQ1_Lost", "DAQ1_Validated"]
        self.daq2keys = ["DAQ2_ID", "DAQ2_TriggerCounts", "DAQ2_Valid", "DAQ2_Flag", "DAQ2_Lost", "DAQ2_Validated"]
        self.ckeys = ["clu_ModuleID", "clu_PlaneID", "clu_SegmentID", "clu_ViewFlag", "clu_SideFlag", "clu_Size", "clu_Charge_HG", "clu_ChargeStd_HG", "clu_Charge_LG", "clu_ChargeStd_LG", "clu_Position_HG"]
        
        self.segment_dict = {"B": 0, "D": 1, "T": 2, "W": 3}
        self.view_dict = {"X": 0, "Y": 1}
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        #self.default_pedestal_gain_files_path = os.path.join(self.base_dir, 'bb', 'B')
        

    
    def load_asic_info(self, asic_info_file):
        asic_info_df = pd.read_csv(asic_info_file)
        self.asic_info = {}
        for _, asic_row in asic_info_df.iterrows():
            feb = asic_row["FEB"]
            daq = asic_row["DAQ"]
            asic = asic_row["ASIC"]
            cable = asic_row["CABLE"]
            ftk_type = asic_row["FTK_TYPE"]
            ftk_module = asic_row["FTK_MODULE"]
            ftk_side = asic_row["FTK_SIDE"]
            connector = asic_row["CONNECTOR"]
        self.asic_info[(feb, daq, asic)] = {"Cable": cable, "FTK_type": ftk_type, "FTK_module": ftk_module, "FTK_side": ftk_side, "Connector": connector}

    def load_array_info(self, array_info_file):
        array_info_df = pd.read_csv(array_info_file)
        self.array_info = {}
        for _, array_row in array_info_df.iterrows():
            ftk_type = array_row["FTK_TYPE"]
            module = array_row["FTK_MODULE"]
            ftk_side = array_row["FTK_SIDE"]
            PlaneID = array_row["PLANE_ID"]
            SegmentID = array_row["SEGMENT_ID"]
            ViewFlag = array_row["VIEW"]
            SideFlag = array_row["SIDE_FLAG"]
            Zposition = array_row["ZPOSITION"]
            self.array_info[(ftk_type, module, ftk_side)] = {"PlaneID": PlaneID, "SegmentID": SegmentID, "ViewFlag": ViewFlag, "SideFlag": SideFlag, "Zposition": Zposition}
            

    def load_data(self, datafile, feb="ZF0", dataversion=1, nevts2read=None):
        if self.data is None:
            self.data = {}
            self.dataversion = dataversion
            self.data[feb] = OpenFileZFEB(datafile, dataversion, nevts2read)
            self.nevts = self.data[feb][list(self.data[feb].keys())[0]].shape[0] # This is the same for multiple febs is they operate in ms mode. If not, different ZirettinoRun objects must be created for the different febs
        else:
            if self.dataversion != dataversion:
                raise ValueError("The dataversion of the data loaded must be the same.")
            self.data[feb] = OpenFileZFEB(datafile, dataversion, nevts2read)  
            if self.nevts != self.data[feb][list(self.data[feb].keys())[0]].shape[0]:
                print("Attenzione: sono state aggiunte due feb con un diverso numero di eventi.")
        self.nfebs = self.nfebs + 1
        if self.gains_dfs is None:
            self.gains_dfs = {}
        if self.pedestals_dfs is None:
            self.pedestals_dfs = {}
        for daq in [1, 2]:
            for gain in ["HG", "LG"]:
                self.pedestals_dfs[(feb, daq, gain)] = pd.read_csv(f"{self.base_dir}/default_pedestal_gain_files/default_pedestals_{gain}.csv")
                self.gains_dfs[(feb, daq, gain)] = pd.read_csv(f"{self.base_dir}/default_pedestal_gain_files/default_gains_{gain}.csv")
        
    def load_configurator(self, configurator_file, feb="ZF0"):
        if self.configurators is None:
            self.configurators = {}
        with open(configurator_file, "r") as file:
            self.configurators[feb] = json.load(file)
         
    def load_pedestals(self, pedestalfile, feb="ZF0", daq=1, gain="HG"):
        if self.pedestals_dfs is None:
            self.pedestals_dfs = {}
        self.pedestals_dfs[(feb, daq, gain)] = pd.read_csv(pedestalfile)

    def subtract_pedestals(self, feb="ZF0", daq=1, gain="HG", sigmacut=0):
        if self.pedestal_subtracted_febs_daqs_gains is None:
            self.pedestal_subtracted_febs_daqs_gains = set()
        self.pedestal_subtracted_febs_daqs_gains.add((feb, daq, gain))
        self.data[feb][f"DAQ{daq}_{gain}_ps"] = self.data[feb][f"DAQ{daq}_{gain}"] - self.pedestals_dfs[(feb, daq, gain)]["pedestal"].to_numpy()
        self.data[feb][f"DAQ{daq}_{gain}_ps"] = np.where(self.data[feb][f"DAQ{daq}_{gain}_ps"] > (sigmacut*self.pedestals_dfs[(feb, daq, gain)]["sigma"].to_numpy()), self.data[feb][f"DAQ{daq}_{gain}_ps"], 0)
        

    def load_gains(self, gainfile, feb="ZF0", daq=1, gain="HG", mode="channel"):
        if self.gains_dfs is None:
            self.gains_dfs = {}
        if mode == "channel":
            self.gains_dfs[(feb, daq, gain)] = pd.read_csv(gainfile)

    def calibrate_charge(self, feb="ZF0", daq=1, gain="HG"):
        if self.charge_calibrated_febs_daqs_gains is None:
            self.charge_calibrated_febs_daqs_gains = set()
        self.charge_calibrated_febs_daqs_gains.add((feb, daq, gain))
        self.data[feb][f"DAQ{daq}_{gain}_pe"] = self.data[feb][f"DAQ{daq}_{gain}_ps"]/self.gains_dfs[(feb, daq, gain)]["gain"].to_numpy()

    def separate_asics(self):
        # Creates a dictionary self.asics_data where the keys are (feb, daq, asic) tuples
        # The values are the arrays of HG, LG, Hit data (and eventually HG_ps, HG_pe, LG_ps, LG_pe) for the channels belonging to the specified asic
        # NOTE: the items of self.asics_data are created EVEN IF the asic is not connected to any SiPM
        
        if self.dataversion == 1:
            asic_keys = ["HG", "LG", "Hit"]
        if self.asics_data is None:
            self.asics_data = {}
        for feb in self.data.keys():
            for daq in [1, 2]:
                for asic in range(4):
                    self.asics_data[(feb, daq, asic)] = {}
                    for key in asic_keys:
                        self.asics_data[(feb, daq, asic)][key] = self.data[feb][f"DAQ{daq}_{key}"][:, 32*asic:32*asic + 32]
                    for gain in ["HG", "LG"]:
                        if (feb, daq, gain) in self.pedestal_subtracted_febs_daqs_gains:
                            self.asics_data[(feb, daq, asic)][gain + "_ps"] = self.data[feb][f"DAQ{daq}_{gain}_ps"][:, 32*asic:32*asic + 32]
                        if (feb, daq, gain) in self.charge_calibrated_febs_daqs_gains:
                            self.asics_data[(feb, daq, asic)][gain + "_pe"] = self.data[feb][f"DAQ{daq}_{gain}_pe"][:, 32*asic:32*asic + 32]


    def reconstruct_ftk_module_geometry(self, ftk="ftk", module=0, psHG=True, peHG=True, psLG=True, peLG=True):
        # Creates a dictionary (1) self.ftk_modules_data, where the keys are (ftk_type, module) tuples
        # The values are again dictionaries (2) where the keys are the ftk_sides (can be either 0, 1, 2, 3)
        # And the (final) values are dictionaries (3) where the keys are variables associated to channels HG, LG, Hit data (and eventually HG_ps, HG_pe, LG_ps, LG_pe)
        # And the values are np.arrays with shape (self.nevts, 96) storing the variables for the 96 channels of the given sidein the events
        # NOTE: If part of the arrays reading a given side (32 channels) are not connected (according to the asic info file), the variables will be set to 0 for the corresponding channels

        asic_keys = ["HG", "LG", "Hit"]
        if psHG == True:
            asic_keys.append("HG_ps")
        if psLG == True:
            asic_keys.append("LG_ps")
        if peHG == True:
            asic_keys.append("HG_pe")
        if peLG == True:
            asic_keys.append("LG_pe")
        
        if self.ftk_modules_data is None:
            self.ftk_modules_data = {}
        self.ftk_modules_data[(ftk, module)] = {}

        
        # According to the asic info, select the asics that are connected to the SiPM arrays reading the different sides of the ftk module considered and append their identifier (feb, daq, asic) to ftk_module_asics list
        ftk_module_asics = []
        for asic_info_key, asic_info in self.asic_info.items():
            if asic_info["FTK_type"] == ftk and asic_info["FTK_module"] == module:
                ftk_module_asics.append(asic_info_key)
                
            
        
        
        for ftk_side in [0, 1, 2, 3]:
            self.ftk_modules_data[(ftk, module)][ftk_side] = {}
            for key in asic_keys:
                self.ftk_modules_data[(ftk, module)][ftk_side][key] = np.zeros((self.nevts, 96))
                
            for iconnector, connector in enumerate(["J1", "J2", "J3"]):
                for asic_info_key in ftk_module_asics:
                    if self.asic_info[asic_info_key]["Connector"] == connector:
                        for key in asic_keys:
                            self.ftk_modules_data[(ftk, module)][ftk_side][key][:, 32*iconnector:32*iconnector + 32] = self.asics_data[asic_info_key][key]
                        break
                else:
                    for key in asic_keys:
                        self.ftk_modules_data[(ftk, module)][ftk_side][key][:, 32*iconnector:32*iconnector + 32] = 0

                
                

    def clustering(self, clukey="HG_pe", thd=6):
        clusters = {}
        for key in self.ckeys:
            clusters[key] = []
        for ftk_module in self.ftk_modules_data.keys():
            for ievt in range(self.nevts):
                if ievt%1000 == 0:
                    print(ievt)
                clusters_event = {}
                for key in self.ckeys:
                    clusters_event[key] = []
                for ftk_side in [0, 1, 2, 3]:
                    hits = np.where(self.ftk_modules_data[ftk_module][ftk_side][clukey][ievt] > thd)[0]
                    if len(hits) == 0: continue
                    breaks = np.where(np.diff(hits) > 1)[0] + 1
                    indices_clusters_event_side = np.split(hits, breaks)
                    for clu in indices_clusters_event_side:
                        clusters_event["clu_ModuleID"].append(ftk_module[1])
                        clusters_event["clu_PlaneID"].append(self.array_info[ftk_module[0], ftk_module[1], ftk_side]["PlaneID"])
                        clusters_event["clu_SegmentID"].append(self.segment_dict[self.array_info[ftk_module[0], ftk_module[1], ftk_side]["SegmentID"]])
                        clusters_event["clu_ViewFlag"].append(self.view_dict[self.array_info[ftk_module[0], ftk_module[1], ftk_side]["ViewFlag"]])
                        clusters_event["clu_SideFlag"].append(self.array_info[ftk_module[0], ftk_module[1], ftk_side]["SideFlag"])
                        clusters_event["clu_Size"].append(len(clu))
                        clusters_event["clu_Charge_HG"].append(np.sum(self.ftk_modules_data[ftk_module][ftk_side]["HG_pe"][ievt][clu]))
                        if clusters_event["clu_Charge_HG"][-1] == 0:
                            print("ZERO CHARGE CLU", ievt)
                            print(indices_clusters_event_side)
                            print("SIDE", ftk_side)
                            print(clukey, self.ftk_modules_data[ftk_module][ftk_side][clukey][ievt])
                            print("indices", indices_clusters_event_side)
                            
                        clusters_event["clu_ChargeStd_HG"].append(np.std(self.ftk_modules_data[ftk_module][ftk_side]["HG_pe"][ievt][clu]))
                        clusters_event["clu_Charge_LG"].append(np.sum(self.ftk_modules_data[ftk_module][ftk_side]["LG_pe"][ievt][clu]))
                        clusters_event["clu_ChargeStd_LG"].append(np.std(self.ftk_modules_data[ftk_module][ftk_side]["LG_pe"][ievt][clu]))
                        
                        clu_avg_stripID_HG = np.average(clu, weights=self.ftk_modules_data[ftk_module][ftk_side]["HG_pe"][ievt][clu])
                        clusters_event["clu_Position_HG"].append(PositionZFTK(clu_avg_stripID_HG, pitch=1., N_sipm_arrays=3, Nstrips1=32, sgn=self.array_info[ftk_module[0], ftk_module[1], ftk_side]["SideFlag"]))
                        #clu_avg_stripID_LG = np.average(clu, weights=self.ftk_modules_data[ftk_module][ftk_side]["LG_pe"][ievt][clu])
                        #clusters_event["Position_LG"].append(PositionZFTK(clu_avg_stripID_LG, pitch=1., N_sipm_arrays=3, Nstrips1=32, sgn=self.array_info[ftk_module[0], ftk_module[1], ftk_side]["SideFlag"]))
                        
                        
                for key in self.ckeys:
                    clusters[key].append(clusters_event[key])
        self.clusters = ak.zip(clusters)


    def save_rootfile(self, rootfile):
        
        with uproot.recreate(rootfile) as fout:
            
            if self.dataversion == 1:
                eventID = np.arange(self.nevts)
                triggerID = self.data[list(self.data.keys())[0]]["TriggerTag"]
                spillID = self.data[list(self.data.keys())[0]]["SpillID"]
                
                
        
                fout_dict =  {"EventID": eventID, "TriggerID": triggerID, "SpillID": spillID, "": self.clusters}                        
                fout["ZDATA"] = fout_dict
            #TO BE IMPLEMENTED: ['NStrip', 'flag_Flag', 'flag_Lost', 'nreco', 'reco_StripNr', 'reco_ADC_HG', 'reco_ADC_LG', 'reco_Npe_HG', 'reco_Npe_LG', 'nclu', ]   
                
            print(f"Saved rootfile at: {rootfile}")
        
                    
                    
                
                
            

                

    

                  

    
                
    
        
        

    

    
    