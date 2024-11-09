import os
import numpy as np
import pandas as pd
import json
import uproot
import awkward as ak
import copy
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
        self.nevts0 = None
        self.configurators = None
        self.pedestals_dfs = None
        self.gains_dfs = None
        self.sync_indices = None
        self.ext_trigger_indices = None
        self.asics_data = None
        self.ftk_modules_data = None
        self.Nstrips1 = 32
        self.N_sipm_arrays = 3
        self.SiPM_pitch = 1
        self.pst_data = None
        self.eventkeys = ["Timestamp", "TriggerTag", "SpillID"]
        self.daq1keys = ["DAQ1_ID", "DAQ1_TriggerCounts", "DAQ1_Valid", "DAQ1_Flag", "DAQ1_Lost", "DAQ1_Validated"]
        self.daq2keys = ["DAQ2_ID", "DAQ2_TriggerCounts", "DAQ2_Valid", "DAQ2_Flag", "DAQ2_Lost", "DAQ2_Validated"]
        self.ckeys = ["ModuleID", "PlaneID", "SegmentID", "ViewFlag", "SideFlag", "FirstStrip", "Size", "Charge_HG", "ChargeStd_HG", "Charge_LG", "ChargeStd_LG", "Position_HG", "PositionError_HG", "Position_LG", "PositionError_LG"]
        self.rkeys = ["StripNr", "ADC_HG", "ADC_LG", "Npe_HG", "Npe_LG"]
        self.segment_dict = {"B": 0, "D": 1, "T": 2, "W": 3}
        self.view_dict = {"X": 0, "Y": 1}
        self.base_dir = os.path.dirname(os.path.abspath(__file__))


    def set_timestamp(self, ts):
        self.timestamp = ts

    
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

    
    def load_pst_mapping(self, pst_mapping_file):
        pst_mapping_df = pd.read_csv(pst_mapping_file)
        pstA_mapping_df = pst_mapping_df.loc[pst_mapping_df["Name"]=="PST_A"]
        pstA_mapping_dict = {}
        for layer in range(1,9):
            pstA_mapping_dict[layer] = {}
            for bar in range(1,4):
                pstA_mapping_dict[layer][bar] = pstA_mapping_df.loc[(pstA_mapping_df["Layer"] == layer) & (pstA_mapping_df["Bar"] == bar)]["Channel"].values[0]
        pstB_mapping_df = pst_mapping_df.loc[pst_mapping_df["Name"]=="PST_B"]
        pstB_mapping_dict = {}
        for layer in range(1,9):
            pstB_mapping_dict[layer] = {}
            for bar in range(1,4):
                pstB_mapping_dict[layer][bar] = pstB_mapping_df.loc[(pstB_mapping_df["Layer"] == layer) & (pstB_mapping_df["Bar"] == bar)]["Channel"].values[0]
        self.pst_mapping = {}
        self.pst_mapping["pstA"] = pstA_mapping_dict
        self.pst_mapping["pstB"] = pstB_mapping_dict            
        
           
    def load_data(self, datafile, feb="ZF0", dataversion=1, nevts2read=None):
        if self.data is None or self.nevts is None:
            self.data = {}
            self.nevts = {}
            self.dataversion = dataversion
            self.data[feb] = OpenFileZFEB(datafile, dataversion, nevts2read)
            self.nevts[feb] = self.data[feb][list(self.data[feb].keys())[0]].shape[0]
            self.nevts0 = -1
        elif feb in self.data.keys():
            print(f"{feb} data already loaded.")
            return
        else:
            if self.dataversion != dataversion:
                raise ValueError("The dataversion of the data loaded must be the same.")
            self.data[feb] = OpenFileZFEB(datafile, dataversion, nevts2read)  
            self.nevts[feb] = self.data[feb][list(self.data[feb].keys())[0]].shape[0]
            self.nevts0 = -1
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

    
    def suppress_non_validated(self):
        for feb in self.febs:
            for daq in [1, 2]:
                for gain in ["HG", "LG"]:
                    self.data[feb][f"DAQ{daq}_{gain}"][self.data[feb][f"DAQ{daq}_Validated"] == 0, :] = 0
                self.data[feb][f"DAQ{daq}_Hit"][self.data[feb][f"DAQ{daq}_Validated"] == 0, :] = 0


    def load_sync_indices(self, sync_indices, feb="ZF0"):
        if self.sync_indices is None:
            self.sync_indices = {}
            self.sync_indices[feb] = sync_indices
        else:
            self.sync_indices[feb] = sync_indices
            if len(sync_indices) != len(list(self.sync_indices.values())[0]):
                print("The febs have different number of synchronous indices. Not possible to apply geometry reconstruction!")
                print("If you want to proceed to geometry reconstruction while exploiting one feb at a time, treat the two febs data as separate runs.")

    
    def load_ext_trigger_indices(self, ext_trigger_indices):
        self.ext_trigger_indices = ext_trigger_indices

        
    def synchronize_boards(self):
        for feb in self.sync_indices.keys():
            for key in self.data[feb].keys():
                self.data[feb][key] = self.data[feb][key][self.sync_indices[feb]]
            self.nevts[feb] = self.data[feb][list(self.data[feb].keys())[0]].shape[0]
        
    
    def enable_geometry_reconstruction(self):
        nevts = list(self.nevts.values())
        if all(n == nevts[0] for n in nevts):
            self.nevts0 = nevts[0]
            if self.ext_trigger_indices is not None:
                if len(self.ext_trigger_indices) != self.nevts0:
                    self.ext_trigger_indices = np.zeros(self.nevts0)                    
        else:
            print("Geometry reconstruction not enabled: different number of events for the febs.")
            for feb in self.febs:
                print(f"Number of events for {feb}: {self.nevts[feb]}")
            self.nevts0 = -1
        
                
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
        
        if self.dataversion == 1 or self.dataversion == 0:
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
        # And the values are np.arrays with shape (self.nevts0, 96) storing the variables for the 96 channels of the given sidein the events
        # NOTE: If part of the arrays reading a given side (32 channels) are not connected (according to the asic info file), the variables will be set to 0 for the corresponding channels

        if self.nevts0 is None or self.nevts0 < 0:
            raise ValueError("Geometry reconstruction is not enabled.")
            return

        asic_keys = ["HG", "LG", "Hit", "HG_ps", "LG_ps", "HG_pe", "LG_pe"]
        
        if self.ftk_modules_data is None:
            self.ftk_modules_data = {}
        self.ftk_modules_data[(ftk, module)] = {}
        # According to the asic info, select the asics that are connected to the SiPM arrays reading the different sides of the ftk module considered and append their identifier (feb, daq, asic) to ftk_module_asics list
        ftk_module_asics = []
        for asic_info_key, asic_info in self.asic_info.items():
            if asic_info["FTK_type"] == ftk and asic_info["FTK_module"] == module:
                ftk_module_asics.append(asic_info_key) #asic_info_key = (feb, daq, asic)
                
        for ftk_side in [0, 1, 2, 3]:
            
            # Initialize ftk_modules_data for the given side to zeros (for any of the asic_keys)
            self.ftk_modules_data[(ftk, module)][ftk_side] = {}
            for key in asic_keys:
                self.ftk_modules_data[(ftk, module)][ftk_side][key] = np.zeros((self.nevts0, 96))
                
            for iconnector, connector in enumerate(["J1", "J2", "J3"]):
                print(ftk_side, connector)
                for asic_info_key in ftk_module_asics:
                    if self.asic_info[asic_info_key]["Connector"] == connector and self.asic_info[asic_info_key]["FTK_side"] == ftk_side and asic_info_key in self.asics_data.keys():
                        print("Found", asic_info_key)
                        for key in asic_keys:
                            self.ftk_modules_data[(ftk, module)][ftk_side][key][:, 32*iconnector:32*iconnector + 32] = self.asics_data[asic_info_key][key]
                        break
                else:
                    # Thanks to this for/break/else construct, even if the data of the asic that should read a side/connector were not loaded, the corrisponding channel data are set to 0 (and the geometry reconstruction continues)
                    for key in asic_keys:
                        self.ftk_modules_data[(ftk, module)][ftk_side][key][:, 32*iconnector:32*iconnector + 32] = 0

    
    def reconstrunct_pst_geometry(self, pst="pstA"):

        if self.nevts0 is None or self.nevts0 < 0:
            raise RunTimeError("Geometry reconstruction is not enabled.")
            return
            
        for asic_info_key, asic_info in self.asic_info.items():
            if asic_info["FTK_type"] == pst:
                pst_module_asic = asic_info_key #asic_info_key = (feb, daq, asic)
                break
        else:
            print(f"{pst} was not used.")
            return
        if self.pst_data is None:
            self.pst_data = {}
        self.pst_data[pst] = {}
        asic_keys = ["HG", "LG", "Hit", "HG_ps", "LG_ps", "HG_pe", "LG_pe"]
        for key in asic_keys:
            self.pst_data[pst][key] = np.zeros((self.nevts0, 8, 3))
            for layer in range(8):
                for bar in range(3):
                    self.pst_data[pst][key][:, layer, bar] = self.asics_data[asic_info_key][key][:, self.pst_mapping[pst][layer + 1][bar + 1]]
                    
    
    def reconstruction(self, recokey="HG_pe", thd=6):
        recos = {}
        for ftk_module in self.ftk_modules_data.keys():
            for ftk_side in [0, 1, 2, 3]:

                ftk_module_data_side = ak.drop_none(ak.Array(self.ftk_modules_data[ftk_module][ftk_side])) # This line transforms the dictionary of ndarrays into a highlevel awkward jagged array               
                hits = ftk_module_data_side[recokey] > thd
                if ftk_side == 0:
                    recos["StripNr"] = ak.local_index(hits)[hits] + ftk_side*96
                    recos["ADC_HG"] = ftk_module_data_side["HG_ps"][hits]
                    recos["ADC_LG"] = ftk_module_data_side["LG_ps"][hits]
                    recos["Npe_HG"] = ftk_module_data_side["HG_pe"][hits]
                    recos["Npe_LG"] = ftk_module_data_side["LG_pe"][hits]
                else:
                    recos["StripNr"] = ak.concatenate([recos["StripNr"], ak.local_index(hits)[hits]], axis=1)
                    recos["ADC_HG"] = ak.concatenate([recos["ADC_HG"], ftk_module_data_side["HG_ps"][hits]], axis=1)
                    recos["ADC_LG"] = ak.concatenate([recos["ADC_LG"], ftk_module_data_side["LG_ps"][hits]], axis=1)
                    recos["Npe_HG"] = ak.concatenate([recos["Npe_HG"], ftk_module_data_side["HG_pe"][hits]], axis=1)
                    recos["Npe_LG"] = ak.concatenate([recos["Npe_LG"], ftk_module_data_side["LG_pe"][hits]], axis=1)
        self.recos = ak.zip(recos)
        

    def clustering(self, clukey="HG_pe", thd=6):
        clusters = {}
        for key in self.ckeys:
            clusters[key] = []
        for ftk_module in self.ftk_modules_data.keys():
            print(f"Starting clustering for FTK: {ftk_module}")
            for ievt in range(self.nevts0):
                if ievt%1000 == 0:
                    print(f"At event: {ievt}/{self.nevts0}")
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
                        clusters_event["FirstStrip"].append(clu[0])
                        clusters_event["Size"].append(len(clu))
                        clusters_event["Charge_HG"].append(np.sum(self.ftk_modules_data[ftk_module][ftk_side]["HG_pe"][ievt][clu]))
                        if clusters_event["Charge_HG"][-1] == 0:
                            raise ValueError(f"FOUND A ZERO CHARGE CLUSTER AT EVENT: {ievt}")
                            
                        clusters_event["ChargeStd_HG"].append(np.std(self.ftk_modules_data[ftk_module][ftk_side]["HG_pe"][ievt][clu]))
                        clusters_event["Charge_LG"].append(np.sum(self.ftk_modules_data[ftk_module][ftk_side]["LG_pe"][ievt][clu]))
                        clusters_event["ChargeStd_LG"].append(np.std(self.ftk_modules_data[ftk_module][ftk_side]["LG_pe"][ievt][clu]))
                        
                        clu_avg_stripID_HG = np.average(clu, weights=self.ftk_modules_data[ftk_module][ftk_side]["HG_pe"][ievt][clu])
                        clusters_event["Position_HG"].append(PositionZFTK(clu_avg_stripID_HG, pitch=self.SiPM_pitch, N_sipm_arrays=self.N_sipm_arrays, Nstrips1=self.Nstrips1, sgn=self.array_info[ftk_module[0], ftk_module[1], ftk_side]["SideFlag"]))
                        
                        clusters_event["PositionError_HG"].append(self.SiPM_pitch/np.sqrt(12)/clusters_event["Charge_HG"][-1]*np.sqrt(np.sum(np.power(self.ftk_modules_data[ftk_module][ftk_side]["HG_pe"][ievt][clu],2))))
                        
                        if np.sum(self.ftk_modules_data[ftk_module][ftk_side]["LG_pe"][ievt][clu]) != 0:
                            clu_avg_stripID_LG = np.average(clu, weights=self.ftk_modules_data[ftk_module][ftk_side]["LG_pe"][ievt][clu])
                            clusters_event["Position_LG"].append(PositionZFTK(clu_avg_stripID_LG, pitch=self.SiPM_pitch, N_sipm_arrays=self.N_sipm_arrays, Nstrips1=self.Nstrips1, sgn=self.array_info[ftk_module[0], ftk_module[1], ftk_side]["SideFlag"]))
                            clusters_event["PositionError_LG"].append(self.SiPM_pitch/np.sqrt(12)/np.sum(clusters_event["Charge_LG"][-1])*np.sqrt(np.sum(np.power(self.ftk_modules_data[ftk_module][ftk_side]["LG_pe"][ievt][clu],2))))
                        else:
                            clusters_event["Position_LG"].append(-999)
                            clusters_event["PositionError_LG"].append(-999)
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
                    eventID[feb] = np.arange(self.nevts0)
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
                    fout_dict.update({f"{feb}_EventID": eventID[feb], f"{feb}_TriggerID": triggerID[feb], f"{feb}_SpillID": spillID[feb], f"{feb}": daqkeys[feb], f"Ext_TriggerID": self.ext_trigger_indices})
                fout_dict["clu"] = self.clusters
                fout_dict["reco"] = self.recos               
                fout["ZDATA"] = fout_dict
                print(f"Saved rootfile at: {rootfile}")
                print("The tree has the following branches:")
        rf = uproot.open(rootfile)
        tree = rf["ZDATA"]
        print(tree.arrays().fields)


    def __add__(self, zrun1):
        if isinstance(zrun1, ZirettinoRun):
            mzrun = copy.copy(self)
            if mzrun.data is not None and zrun1.data is not None:
                for feb in mzrun.data.keys():
                    for key in mzrun.data[feb].keys():
                        mzrun.data[feb][key] = np.concatenate([mzrun.data[feb][key], zrun1.data[feb][key]], axis=0)
                mzrun.nevts += zrun1.nevts
                print("Data concatenated.")
            '''
            if mzrun.asics_data is not None and zrun1.asics_data is not None:
                mzrun.asics_data = np.concatenate([mzrun.asics_data, zrun1.asics_data], axis=0)
                print("Concatenating asics_data.")
            if mzrun.ftk_modules_data is not None and zrun1.ftk_modules_data is not None:
                mzrun.ftk_modules_data = np.concatenate([mzrun.ftk_modules_data, zrun1.ftk_modules_data], axis=0)
                print("Concatenating ftk_modules_data.")
            if mzrun.pst_data is not None and zrun1.pst_data is not None:
                mzrun.pst_data = np.concatenate([mzrun.pst_data, zrun1.pst_data], axis=0)
                print("Concatenating pst_data.")
            '''
            return mzrun
        return NotImplemented  
        
                    
                    
                
                
            

                

    

                  

    
                
    
        
        

    

    
    