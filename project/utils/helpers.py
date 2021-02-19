import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import torch
import torch.nn.functional as F
import sqlite3
import hashlib

# Taken from 'cardiomypathy/utils/waveform.py'
SUBJECT_LEADS_LABELS = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
WAVEFORM_DIR = '/home/jupyter-jeroen/rhythm_python'
MEDIAN_WAVEFORM_DIR = '/home/jupyter-jeroen/median_python'

def generate_path(pseudo_id):
    """ Generates path with the following structure: ps/eu/doid """
    pseudo_id = str(pseudo_id)
    return pseudo_id[0:2] + '/' + pseudo_id[2:4] + '/' + pseudo_id[4:-1]


def get_waveform_path(pseudo_id, test_id, waveform_dir=WAVEFORM_DIR):
    return os.path.join(waveform_dir, generate_path(int(pseudo_id)), f'{int(test_id)}.npy')


def get_waveform(pseudo_id, test_id):
    waveform = np.load(get_waveform_path(pseudo_id, test_id))
    return waveform


def get_waveform_paths(samples, waveform_dir=WAVEFORM_DIR):
    n_samples = len(samples)
    return [
        get_waveform_path(samples['PseudoID'].iloc[i], samples['TestID'].iloc[i], waveform_dir)
            for i in range(n_samples)
    ]


def lowvoltage_check(df, median_waveform_dir=MEDIAN_WAVEFORM_DIR):
    """For all tests in a dataframe check whether it has a low voltage.
    """
    lowvolts = np.zeros((len(df)), dtype=int)
    extremityvoltage = np.zeros((len(df)))
    precordialvoltage = np.zeros((len(df)))
    for idx in tqdm_notebook(range(len(df))):
        raw_wvf = np.load(get_waveform_path(df['PseudoID'].iloc[idx], df['TestID'].iloc[idx], median_waveform_dir))
        # Get interval of values between QOnset and QOnset+QRSDuration
        qonset, qrsdur = df['QOnset'].iloc[idx].astype(int), df['QRSDuration'].iloc[idx].astype(int)
        # If qonset or qrsdur is lower than 0 (because we converted to int was a NaN or null value in
        # the database) we discard this sample
        if qonset < 1 or qrsdur < 2:
            continue
        # Scale the QRS duration to index range (so divide duration by sampling factor e.g. 1000/500)
        samplebase = df['SampleBase'].iloc[idx].astype(int)
        if samplebase != 500:
            continue
        qrsdur = int(qrsdur/(1000/samplebase))
        qrs = raw_wvf[:, qonset:qonset+qrsdur]
        qrs_max = qrs.max(axis=1)
        qrs_min = qrs.min(axis=1)
        maxvals_per_lead = qrs_max + np.absolute(qrs_min)
        maxampl_per_lead = maxvals_per_lead*0.00488
        # I and II must be lower than or equal to 0.5
        c1 = maxampl_per_lead[0:2].max() <= 0.5
        # V1,V2,V3,V4,V5,V6 must be lower than or equal to 1.0
        c2 = maxampl_per_lead[2:].max() <= 1.0
        if c1 or c2:
            lowvolts[idx] = 1
        extremityvoltage[idx] = maxampl_per_lead[0:2].max()
        precordialvoltage[idx] = maxampl_per_lead[2:].max()
    df['LowVoltage'] = lowvolts
    df['ExtremityVoltage'] = extremityvoltage
    df['PrecordialVoltage'] = precordialvoltage
    return df
  

def to12lead(waveform):
    out = np.zeros((12, waveform.shape[1]))
    out[0:2,:] = waveform[0:2,:] # I and II
    out[2,:] = waveform[1,:] - waveform[0,:] # III = II - I
    out[3,:] = -(waveform[0,:] + waveform[1,:])/2 # aVR = -(I + II)/2
    out[4,:] = waveform[0,:] - (waveform[1,:]/2) # aVL = I - II/2
    out[5,:] = waveform[1,:] - (waveform[0,:]/2) # aVF = II - I/2
    out[6:12,:] = waveform[2:8,:] # V1 to V6
    return out


def plot_waveform(
        lead, raw_wvf, axes, factor=0.6, smooth=True, colormap_style="PuOr", 
        alpha=0.8, title="", xlabel=False, ylabel=False, example=None,
        time_incr=0.002,
    ):
    
    x_length = raw_wvf.shape[1]
    x = np.arange(0, x_length*time_incr, time_incr)

    y2 = raw_wvf[lead, :] * 0.00488
    
    high = 450 * 0.00488
    low = 450 * 0.00488
    extent = [0, max(x), -low, high]
    axes.grid(linestyle = ':', which = "both")
    axes.set_xticks(np.arange(0, 10.1, 0.4))
    axes.set_xticks(np.arange(0, 10.1, 0.2), minor = True)
    axes.set_xlim(extent[0], extent[1])
    axes.set_yticks(np.arange(-2, 10.1, 1))
    axes.set_yticks(np.arange(-2, 10.1, 0.5), minor = True)
    axes.set_ylim(extent[2], extent[3])

    if xlabel:
        axes.set_xlabel('Time (s)')
    if ylabel:
        axes.set_ylabel(ylabel)
    if example:
        axes.text(-0.7, 0, example, fontsize=15, va="center")
    axes.plot(x,y2,color='k',linewidth=2)
    axes.set_title(title)
    axes.set_aspect(1/2.5)
    
    return axes


def plot_leads(raw_wvf, subject_leads=range(8)):
    assert len(subject_leads) > 1, 'Number of leads must be greater than 1'
    fig, axes = plt.subplots(nrows=len(subject_leads), sharex=True, figsize=(20,3.5*len(subject_leads)))
    for i, lead in enumerate(subject_leads):
        xlabel = False
        if i == 7:
            xlabel = True
        plot_waveform(lead, raw_wvf, axes[i], example=SUBJECT_LEADS_LABELS[lead], ylabel='Voltage (mV)', xlabel=xlabel)
    fig.tight_layout()

    return fig


def plot_leads_from_paths(waveform_paths, subject_leads=range(8)):
    for index in range(len(waveform_paths)):
        raw_wvf = np.load(waveform_paths[index])
        plot_leads(raw_wvf)


def plot_leads_qrs(samples, subject_leads=range(8), waveform_dir=WAVEFORM_DIR):
    subject_leads_labels = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    waveform_paths = get_waveform_paths(samples, waveform_dir)
    for index in range(len(waveform_paths)):
        qonset = samples['QOnset'].iloc[index]
        qrsdur = samples['QRSDuration'].iloc[index]
        samplebase = samples['SampleBase'].iloc[index].astype(int)
        fig, axes = plt.subplots(nrows=len(subject_leads), sharex=True, figsize=(20,3.5*len(subject_leads)))
        raw_wvf = np.load(waveform_paths[index])
        factor = 1000/samplebase
        time_incr = 1./samplebase
        for i, lead in enumerate(subject_leads):
            xlabel = False
            if i == 7:
                xlabel = True
            plot_waveform(
                lead, raw_wvf, axes[i], example=subject_leads_labels[lead], ylabel='Voltage (mV)', xlabel=xlabel,
                time_incr=time_incr,
            )
            axes[i].axvline(x=qonset*time_incr)
            axes[i].axvline(x=qonset*time_incr+qrsdur/1000)
        fig.tight_layout()

# Converts UMCTriage categories to expert categories in umcu_validation_set
def convert_predictions_to_expert_categories(p_i, split_expert_test_classes):
    
    predicted_categories = p_i.argmax(dim=1).cpu()
    converted_predictions = torch.zeros(p_i.shape[0]).type(torch.LongTensor)
    if split_expert_test_classes == False:
        converted_logits = torch.zeros(p_i.shape[0], 4).type(torch.FloatTensor)
        converted_predictions[predicted_categories == 0] = 0
        converted_predictions[predicted_categories == 1] = 1
        converted_predictions[predicted_categories == 2] = 2
        converted_predictions[predicted_categories == 3] = 3
        converted_predictions[predicted_categories == 4] = 3
        converted_predictions[predicted_categories == 5] = 1 
        
        # Convert predictions according to:
        # train/test class -> expert class
        # 0 -> 0
        # max (1, 5) -> 1
        # 2 -> 2
        # max (3, 4) -> 3
        
        converted_logits[:, 0] = p_i[:, 0]
        converted_logits[:, 1] = torch.max(p_i[:, 1], p_i[:, 5])
        converted_logits[:, 2] = p_i[:, 2]
        converted_logits[:, 3] = p_i[:, 3:5].max(dim=1).values
        
    # Triage category 4 (acute) split into 3 and 4
    elif split_expert_test_classes == True:
        converted_logits = torch.zeros(p_i.shape[0], 5).type(torch.FloatTensor)
        
        converted_predictions[predicted_categories == 0] = 0
        converted_predictions[predicted_categories == 1] = 1
        converted_predictions[predicted_categories == 2] = 2
        converted_predictions[predicted_categories == 3] = 3
        converted_predictions[predicted_categories == 4] = 4
        converted_predictions[predicted_categories == 5] = 1 
        
        # Convert predictions according to:
        # train/test class -> expert class
        # 0 -> 0
        # max (1, 5) -> 1
        # 2 -> 2
        # 3 -> 3
        # 4 -> 4
        
        converted_logits[:, 0] = p_i[:, 0]
        converted_logits[:, 1] = torch.max(p_i[:, 1], p_i[:, 5])
        converted_logits[:, 2] = p_i[:, 2]
        converted_logits[:, 3] = p_i[:, 3]
        converted_logits[:, 4] = p_i[:, 4]
        
    
    converted_p_i = converted_logits
    
    return converted_predictions, converted_p_i

# Selects the variance of the predicted category
# This function makes no sense, a torch.gather is enough to get the variances
def convert_variances_to_expert_categories(output2_var, p_i):
    
    predicted_categories = p_i.argmax(dim=1).cpu()
    variances = output2_var[:, predicted_categories]
    
    return variances


################################################################################

def make_dict_from_tree(element_tree):
    """Traverse the given XML element tree to convert it into a dictionary.
 
    :param element_tree: An XML element tree
    :type element_tree: xml.etree.ElementTree
    :rtype: dict
    """
    def internal_iter(tree, accum):
        """Recursively iterate through the elements of the tree accumulating
        a dictionary result.
 
        :param tree: The XML element tree
        :type tree: xml.etree.ElementTree
        :param accum: Dictionary into which data is accumulated
        :type accum: dict
        :rtype: dict
        """
        if tree is None:
            return accum
        
        if tree.getchildren():
            accum[tree.tag] = {}
            for each in tree.getchildren():
                result = internal_iter(each, {})
                if each.tag in accum[tree.tag]:
                    if not isinstance(accum[tree.tag][each.tag], list):
                        accum[tree.tag][each.tag] = [
                            accum[tree.tag][each.tag]
                        ]
                    accum[tree.tag][each.tag].append(result[each.tag])
                else:
                    accum[tree.tag].update(result)
        else:
            accum[tree.tag] = tree.text
        return accum
    return internal_iter(element_tree, {})

def generate_path_sql(pseudo_id, test_id):
    """ Generates path with the following structure: ps/eu/doid """
    pseudo_id = str(pseudo_id)
    test_id = str(test_id)
    return pseudo_id[0:2] + '/' + pseudo_id[2:4] + '/' + pseudo_id[4:-1] + '/' + test_id + '.npy'

class database_connection:
    """ Initializes connection to a SQLite database with name db_name."""
    def __init__(self, db_name):
        # Connect to database
        self.db_conn = sqlite3.connect(db_name)
        self.db_c = self.db_conn.cursor()
        
    def commit_database(self):
        self.db_conn.commit()
        
    def disconnect_database(self):
        self.db_conn.close()
        
    def get_pseudo_id(self, test_id):
        self.db_c.execute('SELECT PseudoID FROM TestDemTable WHERE TestID = ?;',(test_id,))
        self.pseudo_id = self.db_c.fetchall()
    
    def get_path(self, test_id):
        return generate_path(self.pseudo_id) + '/' + test_id + '.npy'
    
    def get_info(self, test_id):
        self.db_c.execute('SELECT PseudoID FROM TestDemTable WHERE PseudoID = ?;',(pseudo_id,))
        
    def insert_row(self, tablename, rec, foreign, id):
        if foreign == 'PseudoID':
            rec['PseudoID'] = id
        if foreign == 'TestID':
            rec['TestID'] = id
        if foreign == 'Both':
            rec['TestID'] = id[0]
            rec['PseudoID'] = id[1]
        keys = ','.join(rec.keys())
        question_marks = ','.join(list('?'*len(rec)))
        self.db_c.execute('INSERT INTO ' + tablename + '('+keys+') VALUES ('+question_marks+')', tuple(rec.values()))

 # Create results directory if it does not exist
def create_results_directory():
    if not os.path.exists('results/'):
        os.makedirs('results/')

 # Create weights directory if it does not exist
def create_weights_directory():
    if not os.path.exists('weights/'):
        os.makedirs('weights/')

