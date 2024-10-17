import os
import numpy as np
import pandas as pd
import json
import uproot
import awkward as ak
from zirettino_open_file import *
from zirettino_useful_functions import *


class ZirettinoRun():

    def __init__(self):
        self.timestamp = None
        self.dataversion = None
        self.asic_info = None
        self.array_info = None
        self.febs = []
        self.nfebs = 0
        self.data = None
        self.nevts = None
        self.configurators = None
        self.pedestals_dfs = None
        self.gains_dfs = None
        self.asics_data = None
        self.ftk_modules_data = None
        #self.ckeys = ["clu_ModuleID", "clu_PlaneID", "clu_SegmentID", "clu_ViewFlag", "clu_SideFlag", "clu_FirstStrip", "clu_Size", "clu_Charge_HG", "clu_ChargeStd_HG", "clu_Charge_LG", "clu_ChargeStd_LG", "clu_Position_HG", "clu_PositionError_HG", "clu_Position_LG", "clu_PositionError_LG"]
        self.eventkeys = ["Timestamp", "TriggerTag", "SpillID"]
        self.daq1keys = ["DAQ1_ID", "DAQ1_TriggerCounts", "DAQ1_Valid", "DAQ1_Flag", "DAQ1_Lost", "DAQ1_Validated"]
        self.daq2keys = ["DAQ2_ID", "DAQ2_TriggerCounts", "DAQ2_Valid", "DAQ2_Flag", "DAQ2_Lost", "DAQ2_Validated"]
        self.ckeys = ["ModuleID", "PlaneID", "SegmentID", "ViewFlag", "SideFlag", "Size", "Charge_HG", "ChargeStd_HG", "Charge_LG", "ChargeStd_LG", "Position_HG"]
        
        self.segment_dict = {"B": 0, "D": 1, "T": 2, "W": 3}
        self.view_dict = {"X": 0, "Y": 1}
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        

    
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
            # self.nevts is the same for multiple febs is they operate in ms mode. If not, different ZirettinoRun objects must be created for the different febs
            self.nevts = self.data[feb][list(self.data[feb].keys())[0]].shape[0] 
        else:
            if self.dataversion != dataversion:
                raise ValueError("The dataversion of the data loaded must be the same.")
            self.data[feb] = OpenFileZFEB(datafile, dataversion, nevts2read)  
            if self.nevts != self.data[feb][list(self.data[feb].keys())[0]].shape[0]:
                print("Attenzione: sono state aggiunte due feb con un diverso numero di eventi.")
        self.febs.append(feb)
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

    def subtract_pedestals(self, sigmacut=5):
        for feb in self.febs:
            for daq in [1, 2]:
                for gain in ["HG", "LG"]:
                    self.data[feb][f"DAQ{daq}_{gain}_ps"] = self.data[feb][f"DAQ{daq}_{gain}"] - self.pedestals_dfs[(feb, daq, gain)]["pedestal"].to_numpy()
                    self.data[feb][f"DAQ{daq}_{gain}_ps"] = np.where(self.data[feb][f"DAQ{daq}_{gain}_ps"] > (sigmacut*self.pedestals_dfs[(feb, daq, gain)]["sigma"].to_numpy()), self.data[feb][f"DAQ{daq}_{gain}_ps"], 0)
    

    def load_gains(self, gainfile, feb="ZF0", daq=1, gain="HG", mode="channel"):
        if self.gains_dfs is None:
            self.gains_dfs = {}
        if mode == "channel":
            self.gains_dfs[(feb, daq, gain)] = pd.read_csv(gainfile)

    def calibrate_charge(self):
        for feb in self.febs:
            for daq in [1, 2]:
                for gain in ["HG", "LG"]:
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
                        self.asics_data[(feb, daq, asic)][gain + "_ps"] = self.data[feb][f"DAQ{daq}_{gain}_ps"][:, 32*asic:32*asic + 32]
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
            print("SIDE", ftk_side)
            self.ftk_modules_data[(ftk, module)][ftk_side] = {}
            for key in asic_keys:
                self.ftk_modules_data[(ftk, module)][ftk_side][key] = np.zeros((self.nevts, 96))
                
            for iconnector, connector in enumerate(["J1", "J2", "J3"]):
                print("-----------connector", connector)
                for asic_info_key in ftk_module_asics:
                    print("asic", asic_info_key)
                    if self.asic_info[asic_info_key]["Connector"] == connector and self.asic_info[asic_info_key]["FTK_side"] == ftk_side:
                        print("connected to asic", asic_info_key)
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
                        clusters_event["ModuleID"].append(ftk_module[1])
                        clusters_event["PlaneID"].append(self.array_info[ftk_module[0], ftk_module[1], ftk_side]["PlaneID"])
                        clusters_event["SegmentID"].append(self.segment_dict[self.array_info[ftk_module[0], ftk_module[1], ftk_side]["SegmentID"]])
                        clusters_event["ViewFlag"].append(self.view_dict[self.array_info[ftk_module[0], ftk_module[1], ftk_side]["ViewFlag"]])
                        clusters_event["SideFlag"].append(self.array_info[ftk_module[0], ftk_module[1], ftk_side]["SideFlag"])
                        clusters_event["Size"].append(len(clu))
                        clusters_event["Charge_HG"].append(np.sum(self.ftk_modules_data[ftk_module][ftk_side]["HG_pe"][ievt][clu]))
                        if clusters_event["Charge_HG"][-1] == 0:
                            print("ZERO CHARGE CLU", ievt)
                            print(indices_clusters_event_side)
                            print("SIDE", ftk_side)
                            print(clukey, self.ftk_modules_data[ftk_module][ftk_side][clukey][ievt])
                            print("indices", indices_clusters_event_side)
                            
                        clusters_event["ChargeStd_HG"].append(np.std(self.ftk_modules_data[ftk_module][ftk_side]["HG_pe"][ievt][clu]))
                        clusters_event["Charge_LG"].append(np.sum(self.ftk_modules_data[ftk_module][ftk_side]["LG_pe"][ievt][clu]))
                        clusters_event["ChargeStd_LG"].append(np.std(self.ftk_modules_data[ftk_module][ftk_side]["LG_pe"][ievt][clu]))
                        
                        clu_avg_stripID_HG = np.average(clu, weights=self.ftk_modules_data[ftk_module][ftk_side]["HG_pe"][ievt][clu])
                        clusters_event["Position_HG"].append(PositionZFTK(clu_avg_stripID_HG, pitch=1., N_sipm_arrays=3, Nstrips1=32, sgn=self.array_info[ftk_module[0], ftk_module[1], ftk_side]["SideFlag"]))
                        #clu_avg_stripID_LG = np.average(clu, weights=self.ftk_modules_data[ftk_module][ftk_side]["LG_pe"][ievt][clu])
                        #clusters_event["Position_LG"].append(PositionZFTK(clu_avg_stripID_LG, pitch=1., N_sipm_arrays=3, Nstrips1=32, sgn=self.array_info[ftk_module[0], ftk_module[1], ftk_side]["SideFlag"]))
                        
                        
                for key in self.ckeys:
                    clusters[key].append(clusters_event[key])
        self.clusters = ak.zip(clusters)


    def save_rootfile(self, rootfile):
        
        with uproot.recreate(rootfile) as fout:
            
            
            if self.dataversion == 1:
                eventID = {}
                triggerID = {}
                spillID = {}
                daq1keys = {}
                daq2keys = {}
                daqkeys = {}
                for feb in self.febs:
                    eventID[feb] = np.arange(self.nevts)
                    triggerID[feb] = self.data[feb]["TriggerTag"]
                    spillID[feb] = self.data[feb]["SpillID"]
                    daq1keys[feb] = {}
                    daq2keys[feb] = {}
                    for key in self.daq1keys:
                        daq1keys[feb][key] = self.data[feb][key]
                    for key in self.daq2keys:
                        daq2keys[feb][key] = self.data[feb][key]
                    daqkeys[feb] = daq1keys[feb]
                    daqkeys[feb].update(daq2keys[feb])
                    
                        
                fout_dict = {}
                for feb in self.febs:
                    print(daqkeys)
                    fout_dict.update({f"{feb}_EventID": eventID[feb], f"{feb}_triggerID": triggerID[feb], f"{feb}_spillID": spillID[feb], f"{feb}": daqkeys[feb]})
                fout_dict["clu"] = self.clusters
                    
        
                
                fout["ZDATA"] = fout_dict
            #fout_dict =  {"EventID": eventID, "TriggerID": triggerID, "SpillID": spillID, "": self.clusters}                        
            #TO BE IMPLEMENTED: ['NStrip', 'flag_Flag', 'flag_Lost', 'nreco', 'reco_StripNr', 'reco_ADC_HG', 'reco_ADC_LG', 'reco_Npe_HG', 'reco_Npe_LG', 'nclu', ]   
                
            print(f"Saved rootfile at: {rootfile}")
        
                    
                    
                
                
            

                

    

                  

    
                
    
        
        

    

    
    