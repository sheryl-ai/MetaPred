""" Code for loading data. """
import pandas as pd
import numpy as np
import pickle as pkl
import os, copy, operator
import random

import progressbar
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import *

FLAGS = flags.FLAGS

######################
#### REMOTE FILES ####
######################
f_dxlst = 'Dodge_eirb17110_Diagnoses_Problem_List.csv' #6G
f_dx = 'Dodge_eirb17110_Diagnoses_Encounters.csv' #6G

VISIT_NUM = 0

#####################
#### LOCAL FILES ####
#####################
feature_file = None
dct_file = None

#########################
### DATABASE QUERYING ###
#########################
AD_icd = ['331.0', '331.2', '331.6', '331.7'] # Alzheimer’s Related Disorders
PD_icd = ['332.'] # Parkinson’s Disease
FD_icd = ['331.1', '331.11', '331.19'] # Frontotemporal Dementia
MCI_icd = ['331.83', '331.89', '331.9', '331.90'] # MCI Related Disorders
HD_icd = ['333.4'] # Huntington’s Disease
MO_icd = ['331.3', '331.4', '331.5'] # Mechanical Obstructions
AM_icd = ['780.93'] # Amnesia
DM_icd = ['290.', '291.', '294.', '331.82', '331.83'] # Dementia

root = '~/Documents/data/ohsu/'
file_plist = root + f_dxlst
file_dx = root + f_dx
numlines_plist =35459836
numlines_dx = 32403450


class DataGenerator(object):
    """
    Data Generator capable of generating batches of OHSU data.
    """
    def __init__(self, batch_size, config={}):
        """
        Args:
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.intmd_path = 'intermediate/'
        self.f_pts = 'pts.pkl'
        self.f_dct = 'dct.pkl'
        self.file_list = [file_plist, file_dx]
        self.total_lines= [numlines_plist, numlines_dx]
        self.feature_sets = [['DX_ICD', 'DX_START_DATE', 'DX_END_DATE'], ['ICD9_CODE', 'DX_DATE']]
        self.diseases = {'AD':AD_icd, 'PD':PD_icd, 'FD':FD_icd, 'HD':HD_icd, 'MO':MO_icd, 'MCI':MCI_icd, 'AM':AM_icd, 'DM':DM_icd}
        self.target = 'AD'
        self.min_seq_len = 5
        self.icd2idx = dict()
        self.age_dist = {(55, 60):0, (60, 65):0, (65, 70):0, (70, 75):0, (75, 80):0, (80, 85):0, (85, 90):0, (90, 95):0, (95, 100):0}
        self.read_data()
        self.save_files()

    def read_data(self):
        '''read and store data files into dictionary with matrices as values.
        generate the corresponding controls for the diseases.
        analysis the possible disease prediction tasks.'''
        self.dct, self.small_dct = self.make_dictionary()
        self.codes, self.grp_codes, self.grp_icd2idx = self.process_icd(self.small_dct) # code# 10989, group_codes(list)# 1016, grp_dct(dict)# 1060
        self.X_pos, self.y_pos, self.X_pos_mat, self.y_pos_mat = self.generate_cases()
        self.X_neg, self.y_neg, self.X_neg_mat, self.y_neg_mat = self.generate_controls()
        self.check_patient_set(self.small_dct)


    def make_dictionary(self):
        print("Loading patients ... ")
        chunksize = 10**6
        if os.path.isfile(self.intmd_path + self.f_pts):
            print ('subject id file exits')
            with open(self.intmd_path + self.f_pts, 'rb') as f:
                pts = pkl.load(f)
                f.close()
        else:
            print ('generating subject id file ...')
            pts = []
            for i in range(2): # 2 files needed to be preprocessed
                numlines = self.total_lines[i]
                filename = self.file_list[i]
                cols = list(pd.read_csv(filename, nrows=0).columns)
                for j in range(0, numlines, chunksize):
                    # print(j)
                    df = pd.read_csv(filename, header = None, nrows = chunksize, skiprows = j, names = cols)
                    lst = df.HASH_SUBJECT_ID.unique().tolist()
                    pts = list(set().union(pts, lst))

            os.makedirs(os.path.dirname(self.intmd_path + self.f_pts), exist_ok=True)
            with open(self.intmd_path + self.f_pts,'wb') as f:
                pkl.dump(pts,f,protocol=2)
                f.close()

        dct = {}
        for p in pts: # number of patients
        	dct[p] = {'prob_list':{}, 'dx_history':{}} # two dict per subj: dx history
                                                       # prob list

        print("{0} Patients.".format(len(dct)))

        print("Making big dictionary ... ")
        if os.path.isfile(self.intmd_path + self.f_dct):
            print ('data dictionary file exits')
            with open(self.intmd_path + self.f_dct, 'rb') as f:
                dct = pkl.load(f)
                f.close()
            with open(self.intmd_path + 'small_' + self.f_dct, 'rb') as f:
                small_dct = pkl.load(f)
                f.close()
        else:
            print ('generating data dictionary file ...')
            for i in range(2):
                numlines = self.total_lines[i]
                filename = self.file_list[i]
                features = self.feature_sets[i]
                cols = list(pd.read_csv(filename, nrows=0).columns)
                for j in range(0, numlines, chunksize):
                    df = pd.read_csv(filename, header = None, nrows = chunksize, skiprows = j, names = cols)
                    if i == 0:
                        # pass
                        tmp = df[['HASH_SUBJECT_ID'] + features]
                        tmp = tmp.to_dict('records')
                        for row in tmp:
                            code = row['DX_ICD']
                            subj = row['HASH_SUBJECT_ID']
                            start_t = row['DX_START_DATE']
                            end_t = row['DX_END_DATE']
                            if code in dct[subj]['prob_list'].keys():
                                dct[subj]['prob_list'][code].append((start_t, end_t)) # key code, value time
                            else:
                                dct[subj]['prob_list'][code] = [(start_t, end_t)]
                    elif i==1:
                        tmp = df[['HASH_SUBJECT_ID'] + features]
                        tmp = tmp.to_dict('records')
                        for row in tmp:
                            code = row['ICD9_CODE']
                            date = row['DX_DATE']
                            subj = row['HASH_SUBJECT_ID']
                            if code in dct[subj]['dx_history'].keys():
                                dct[subj]['dx_history'][code].append(date) # key code, value date
                            else:
                                dct[subj]['dx_history'][code] = [date]
            		    # print("number of codes: {0}".format(len(dct[subj]['dx_history'])))
            os.makedirs(os.path.dirname(self.intmd_path + self.f_dct), exist_ok=True)
            with open(self.intmd_path + self.f_dct,'wb') as f:
                pkl.dump(dct,f,protocol=2)
                f.close()

            small_dct = dict([(k, v['dx_history']) for k,v in dct.items()])
            os.makedirs(os.path.dirname(self.intmd_path + 'small_' + self.f_dct), exist_ok=True)
            with open(self.intmd_path + 'small_' + self.f_dct,'wb') as f:
                pkl.dump(small_dct,f,protocol=2)
                f.close()
        print("Dictionary Generated")
        return dct, small_dct

    ####################################
    ### Make X_pos's and X_neg's #######
    ####################################
    def make_pos_features(self, dct, dct_type = 'small',
                        diseases= ['331.0', '331.2', '331.6', '331.7'], age_lim = 99.9,
                      pred_cutoff=0.5, obs_cutoff=2.0,
                      stepsize=1/12, minlen = 6):
        '''note, we are only using dx_history of dct
        diseases: icd9 codes for the disease of interest. default = MCI.
        obs_cutoff: number of years to look back from onset of disease of interest. default = 3 years.
        pred_cutoff: censor window for prediction time. default = 3 months.
        stepsize: t-interval between observations. default = 1/12 years, i.e. 1 month.'''
        global VISIT_NUM
        num_features = len(self.grp_icd2idx)
        if self.target != "PD" and self.target != "DM":
            subj = list(set([s for s in dct.keys() if len(set(dct[s].keys()).intersection(diseases))>=1]))
        else:
            subj = []
            for codes in diseases:
                subj += list(set([s for s in dct.keys() if isfind(codes, set(dct[s].keys()), self.target)]))
            subj = list(set(subj))

        #3394 for AD
        print ('number of patients with the disease: ', len(subj))
        X,y = dict(),dict()

        print("Constructing conditional X ... ")
        print("")
        count = 0
        recs_count, seq_len = [], []
        for iteration in progressbar.progressbar(range(len(subj))):
            s = subj[iteration]
            lst = list(set(dct[s].keys()).intersection(self.codes)) # list of icd-9 codes of the subject, interact with the entire codes set
            if self.target == "PD":
                cd = diseases[0]
                targets = [icd for icd in lst if cd in icd]
            elif self.target == "DM":
                targets = []
                for cd in diseases:
                    if "290." in cd or "291." in cd or "294." in cd:
                        targets += [icd for icd in lst if cd in icd]
                    else:
                        targets += list(set(lst).intersection(set(cd)))
            else:
                targets = list(set(lst).intersection(set(diseases)))    # list of overlaps w/ the targets
            if len(targets)>0: # else: pass
                #get the time ranges for the patients w/ the target disease
                raw_times = [[float(jj) for jj in dct[s][icd]] for icd in targets] # the time records are corresponding to the target point
                # print (raw_times)  i.e., [[97.863, 98.022, 98.266, 98.49700000000001, 97.863]], so the values are ages?
                #obtain starting and end points of observation window
                end = min([starttime[0] for starttime in raw_times]) - pred_cutoff # divide the time into obs_cutoff and pred_cutoff
                                                                                   # the obs_window and pred_window are all before the first diagnosis of the given disease
                start = end - obs_cutoff # start and end are for the observation window
                age_requirement = start < age_lim # which means a reasonable value for age

                #Align the times below:
                timesteps = np.arange(start, end, stepsize) # list of time points, observation window
                # t2idx = dict([(v,k) for k,v in enumerate(timesteps)])
                #Match between obs_window and dataset observations.
                all_times = list(set(flatten([[float(jj) for jj in dct[s][icd]] for icd in lst]))) # consider all the possible timepoints for the patient
                obs_index = np.where(np.logical_and(all_times>=min(timesteps),
                                                      all_times<=max(timesteps)))[0] # overlap of raw_times and timesteps
                obs_times = [all_times[idx] for idx in obs_index]
                t2idx = dict([(v,k) for k,v in enumerate(obs_times)]) # dictionary -- key is time point, value is index
                lengths = len(t2idx)
                if lengths >= 0 and age_requirement:             #>=minlen timesteps, >=2 features
                    features = np.zeros((len(t2idx), num_features))
                    for icd in lst:
                        if icd[0:3] in self.grp_icd2idx.keys():
                            use_times = [float(jj) for jj in dct[s][icd] if (float(jj) <= max(timesteps) and float(jj) >= min(timesteps))]
                            if len(use_times)>0:
                                # indices = np.array(list(set([t2idx[find_nearest(timesteps, t)] \
                                # for t in raw_times])))
                                indices = np.array([t2idx[t] for t in use_times]) # for each valid time point, turn time point into index
                                indices = np.array([[ii, self.grp_icd2idx[icd[0:3]]] for ii in indices])
                                features[indices[:, 0], indices[:,1]] = 1
                    if np.sum(features)>2:
                        recs_count.append(np.sum(features))
                        seq_len.append(len(t2idx))
                        X[s]= features
                        y[s] = 1
                        self.compute_age(start, end)

                        count += 1
                        #### format: x = (feature matrix, list of positive targets) ####
                        # np.save(savefile + s, (features, targets))
        print ("the number of patients of the given disease is: ", count)
        print ("the number of average records per patient is: ", np.mean(np.array(recs_count)))
        print ("the average sequence length is: ", np.mean(np.array(seq_len)))
        VISIT_NUM += np.sum(np.array(seq_len))
        return X,y

    def make_neg_features(self, dct, dct_type = 'small',
                        diseases= ['331.0', '331.2', '331.6', '331.7'], age_lim = 99.9,
                      pred_cutoff=0.5, obs_cutoff=2.0,
                      stepsize=1/12, minlen = 6):
        '''note, we are only using dx_history of dct
        diseases: icd9 codes for the disease of interest. default = MCI.
        obs_cutoff: number of years to look back from onset of disease of interest. default = 3 years.
        pred_cutoff: censor window for prediction time. default = 3 months.
        stepsize: t-interval between observations. default = 1/12 years, i.e. 1 month.'''
        global VISIT_NUM
        # compute age distribution
        age_dist = copy.deepcopy(self.age_dist)
        case_sum = np.sum(np.array(list(self.age_dist.values())))
        for age, num in self.age_dist.items():
            age_dist[age] = np.divide(num, case_sum)
        age_ratio = [value for (key, value) in sorted(age_dist.items())]
        print (age_ratio)

        # compute number of controls
        num_features = len(self.grp_icd2idx)
        subj = []
        for codes in diseases:
            subj += list(set([s for s in dct.keys() if isfind(codes, set(dct[s].keys()), 'controls')]))
        subj = list(set(subj))

        print ('number of patients with the controls: ', len(subj))
        X,y = dict(),dict()

        print("Constructing conditional X ... ")
        print("")
        count, count_obs = 0, 0
        recs_count, seq_len = [], []
        for iteration in progressbar.progressbar(range(len(subj))):
            s = subj[iteration]
            lst = list(set(dct[s].keys()).intersection(self.codes)) # list of icd-9 codes of the subject, interact with the entire codes set
            targets = list(set(lst).intersection(set(diseases)))    # list of overlaps w/ the targets
            if len(lst)>0:
                #get the time ranges for the patients w/ randomly sampled bin
                all_times = list(set(flatten([[float(jj) for jj in dct[s][icd]] for icd in lst])))
                if (max(all_times)-min(all_times)) < obs_cutoff:
                    count_obs += 1
                    continue
                ###
                downcount = 500
                while downcount > 0:
                    bin = np.random.choice(range(1,10), 1, p=age_ratio)[0] # randomly pick a number
                    age_bin_start, age_bin_end = bin * 5 + 50, bin * 5 + 55
                    # bin = np.random.choice(range(1,6), 1, p=age_ratio)[0] # randomly pick a number
                    # age_bin_start, age_bin_end = bin * 10 + 40, bin * 10 + 50
                    if age_bin_end <= min(all_times):
                        downcount  -= 1
                        continue
                    else:
                        end = find_nearest(np.array(all_times), age_bin_end) - pred_cutoff
                        start = end - obs_cutoff # start and end are for the observation window
                        obs_index = np.where(np.logical_and(all_times >= start,
                                                              all_times <= end))[0] # overlap of raw_times and timestep
                        if len(obs_index) >= self.min_seq_len:
                            break
                        else:
                            downcount -= 1
                if downcount == 0:
                    continue
                ### instead of sampling according to age
                # end = max(all_times) - pred_cutoff
                # start = end - obs_cutoff # start and end are for the observation window
                # obs_index = np.where(np.logical_and(np.array(all_times) >= start,
                #                               np.array(all_times) <= end))[0] # overlap of raw_times and timesteps
                ###
                age_requirement = start < age_lim
                lengths = len(obs_index)
                if lengths >= self.min_seq_len and age_requirement:
                    obs_times = [all_times[idx] for idx in obs_index]
                    t2idx = dict([(v,k) for k,v in enumerate(obs_times)]) # dictionary -- key is time point, value is index
                    features = np.zeros((len(t2idx), num_features))
                    for icd in lst:
                        if icd[0:3] in self.grp_icd2idx.keys():
                            raw_times = [float(jj) for jj in dct[s][icd] if (float(jj) <= end and float(jj) >= start)]
                            if len(raw_times)>0:
                                # indices = np.array(list(set([t2idx[find_nearest(timesteps, t)] \
                                # for t in raw_times])))
                                indices = np.array([t2idx[t] for t in raw_times]) # for each valid time point, turn time point into index
                                indices = np.array([[ii, self.grp_icd2idx[icd[0:3]]] for ii in indices])
                                features[indices[:, 0], indices[:,1]] = 1
                    if np.sum(features)>2:
                        recs_count.append(np.sum(features))
                        seq_len.append(len(t2idx))
                        X[s]= features
                        y[s] = 0
                        count += 1

                        #### format: x = (feature matrix, list of positive targets) ####
                        # np.save(savefile + s, (features, targets))
        print ("the number of patients of the controls is: ", count)
        print ("the number of average records per patients is: ", np.mean(np.array(recs_count)))
        print ("the average sequence length is: ", np.mean(np.array(seq_len)))
        VISIT_NUM += np.sum(np.array(seq_len))
        return X,y

    def process_icd(self, dct):
        codes = sorted(list(set(flatten( [list(set(dct[s].keys())) for s in dct] )))) # 10989
        codes = sorted(list(set(codes) - set(codes[9905:10019]))) # 10875
        grp_codes =  sorted(list(set([c[0:3] for c in codes]))) # 1016
        grp_icd2idx = dict([(v,k) for k,v in enumerate(grp_codes)]) # generate a dictionary mapping group code to index, 1016
        self.PADDING_ID = len(grp_icd2idx)
        print ("The PADDING_ID is set as the number of group code, that is ", self.PADDING_ID)
        return codes, grp_codes, grp_icd2idx

    ############################################
    ### Encode index into feature matrix #######
    ############################################
    def encode_Xy(self, X, y):
        '''encode the stored 0-1 value as index of feature dimensions,
        then the size should be (number of sequence x length of feature list)
        '''
        num_features = len(self.grp_icd2idx)
        print (num_features)

        new_X = dict()
        new_y = dict()
        for s in X:
            seq_len, _ = X[s].shape
            tmp_features = np.zeros((seq_len, num_features), dtype="int32") + self.PADDING_ID
            for i in range(seq_len):
                col = 0
                for j in range(num_features):
                    if X[s][i, j] == 1:
                        tmp_features[i, col] = j
                        col += 1
            new_X[s] = tmp_features # i.e., idx1-3 are indices of features , -1 are padding value
                                    # [[idx1, idx2, -1, -1], [idx3, idx4, idx5, -1], [idx3, -1, -1, -1]]
            new_y[s] = y[s]
        return new_X, new_y

    def generate_cases(self):
        '''generate positive samples (disease cases) and labels
        '''
        X_pos_mat, y_pos_mat = self.make_pos_features(self.small_dct, dct_type='small', diseases=self.diseases[self.target])
        X_pos, y_pos = self.encode_Xy(X_pos_mat, y_pos_mat) # turn mat into index
        return X_pos, y_pos, X_pos_mat, y_pos_mat

    def generate_controls(self):
        '''controls should be the subjects with similar diseases (neurodegenerative disease)
        '''
        ctr_diseases_codes = flatten([self.diseases[ctr] for ctr in self.diseases if ctr != self.target]) # list of disease names
        print (ctr_diseases_codes)
        X_neg_mat, y_neg_mat = self.make_neg_features(self.small_dct, dct_type='small', diseases=ctr_diseases_codes)
        X_neg, y_neg = self.encode_Xy(X_neg_mat, y_neg_mat)
        # delete over controls suffering the target disease
        new_X_neg = copy.deepcopy(X_neg)
        new_y_neg = copy.deepcopy(y_neg)
        for s in X_neg:
            if s in self.X_pos:
                del new_X_neg[s]
                del new_y_neg[s]
                del X_neg_mat[s]
                del y_neg_mat[s]
        print ("updated number of controls is: ", len(new_y_neg))
        return new_X_neg, new_y_neg, X_neg_mat, y_neg_mat

    ##############################
    ### Disease statistics #######
    ##############################
    def compute_age(self, start, end):
        for age, num in self.age_dist.items():
            if end > age[0] and end <= age[1]: self.age_dist[age] = num + 1

    def check_patient_set(self, dct):
        global VISIT_NUM
        patients = dict()
        ttl_patients = list()
        for type in self.diseases:
            if type != "PD" and type != "DM":
                subj = list(set([s for s in dct.keys() if len(set(dct[s].keys()).intersection(self.diseases[type]))>=1]))
            else:
                subj = []
                for codes in self.diseases[type]:
                    subj += list(set([s for s in dct.keys() if isfind(codes, set(dct[s].keys()), type)]))
                subj = list(set(subj))
            patients[type] = set(subj)
            ttl_patients += subj
            print("number of patients with the " + type + " disease: ", len(subj))

        for type in self.diseases:
            print ('----------')
            print ("disease type: " + type)
            print ("intersection between " + type + " AND PD: ", len(patients[type].intersection(patients['PD'])))
            print ("intersection between " + type + " AND AD: ", len(patients[type].intersection(patients['AD'])))
            print ("intersection between " + type + " AND HD: ", len(patients[type].intersection(patients['HD'])))
            print ("intersection between " + type + " AND MCI: ", len(patients[type].intersection(patients['MCI'])))
            print ("intersection between " + type + " AND MO: ", len(patients[type].intersection(patients['MO'])))
            print ("intersection between " + type + " AND FD: ", len(patients[type].intersection(patients['FD'])))
            print ("intersection between " + type + " AND AM: ", len(patients[type].intersection(patients['AM'])))
            print ("intersection between " + type + " AND DM: ", len(patients[type].intersection(patients['DM'])))
        print ("number of total visit for the domain: ", VISIT_NUM)
        print ("The total number of neurodegenertive disease are: ", len(set(ttl_patients)))

    def save_files(self):
        print ("save files")
        print ("patients ...")
        f = open(self.intmd_path + self.target + '.pos_.pkl', 'wb')
        pkl.dump((self.X_pos, self.y_pos), f, protocol=2)
        f.close()
        print ("controls ...")
        f = open(self.intmd_path + self.target + '.neg_.pkl', 'wb')
        pkl.dump((self.X_neg, self.y_neg), f, protocol=2)
        f.close()
        print ("then, save 2 very large matrices ...")
        print ("cases ...")
        f = open(self.intmd_path + self.target + '.pos.mat_.pkl', 'wb')
        pkl.dump((self.X_pos_mat, self.y_pos_mat), f, protocol=2)
        f.close()
        print ("controls ...")
        f = open(self.intmd_path + self.target + '.neg.mat_.pkl', 'wb')
        pkl.dump((self.X_neg_mat, self.y_neg_mat), f, protocol=2)
        f.close()
        print ("all files saved!")

def main():
    DataGenerator(32)

if __name__ == "__main__":
    main()
