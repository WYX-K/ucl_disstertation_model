from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
import pandas as pd
import numpy as np
import pickle


def data_preprocessing(file_name):
    """Preprocess the data

    Parameters
    ----------
    file_name: the name of the file containing the data
    """
    data = pd.read_csv(file_name)
    data['StartTime'] = pd.to_datetime(
        data['StartTime'], format='%d/%m/%Y %H:%M')
    data['EndTime'] = pd.to_datetime(data['EndTime'], format='%d/%m/%Y %H:%M')
    data['Pathway'] = (data['Pathway'] != 'Walkin').astype('category')
    max_sequence = max(data.groupby('SpellID_Anon').size())
    area_time_sequence_head = [item for sublist in [
        [x, 'Time'+str(x)] for x in range(1, max_sequence + 1)] for item in sublist]
    col = ['SpellID_Anon', 'Pathway', 'FirstTimetoED'] + \
        area_time_sequence_head+['isAdmitted']

    patients = pd.DataFrame(columns=col)

    # encode the first time to ED
    patients[['SpellID_Anon', 'Pathway', 'FirstTimetoED']
             ] = data[data['AreaSequence'] == 1][['SpellID_Anon', 'Pathway', 'StartTime']]
    patients['FirstTimetoED'] = patients['FirstTimetoED'].apply(
        lambda x: x.hour+x.minute/60)

    # encode the admitted status
    transfer_status_list = pd.Series(data['TransferStatus'].unique())
    uch_list = transfer_status_list[~transfer_status_list.isin(
        ['Arrival-Discharge', 'Arrival', 'Discharged', 'Transfer'])]
    patient_admitted_list = data[data['TransferStatus'].isin(uch_list)]
    patients['isAdmitted'] = patients['SpellID_Anon'].isin(
        patient_admitted_list['SpellID_Anon']).astype(int)

    # encode the area sequence
    with open('dict_area.pkl', 'rb') as f:
        dict_area = pickle.load(f)
    area_list = list(dict_area.keys())
    data_grouped = data.groupby('SpellID_Anon')[
        ['AreaSequence', 'CurrentArea', 'TotalMins']].agg(list)
    data_dict = data_grouped.to_dict('index')

    def add_row(row):
        col_list = data_dict[row['SpellID_Anon']]['AreaSequence']
        patient = data_dict[row['SpellID_Anon']]
        for i in range(0, len(col_list)):
            currentArea = patient['CurrentArea'][i]
            TotalMins = patient['TotalMins'][i]
            row[col_list[i]] = area_list.index(currentArea)
            row['Time'+str(col_list[i])] = TotalMins
        return row

    patients = patients.apply(add_row, axis=1)
    region_cols = [i for i in range(1, max_sequence + 1)]
    time_cols = ['Time' + str(i) for i in range(1, max_sequence + 1)]

    """
    This code is designed to handle rows in a DataFrame where a region sequence (represented by `region_arr`) has a value of 0,
    which is considered an "off-the-floor" (OTF) event.
    The time spent in these OTF events is added to the time spent in the previous region,
    and the OTF event is then removed from the sequence.
    The region and time sequences are then extended with zeros to match the original length.
    """
    def delete_OTF(row):
        # Extract the region and time arrays from the row
        region_arr, time_arr = row[region_cols].values, row[time_cols].values
        # Create a mask where the region array is equal to 0
        mask = region_arr == 0
        # Find the positions where the mask is True (i.e., where the region array is 0)
        zero_positions = np.where(mask)[0]
        # Add the time at the zero positions to the time at the previous positions
        time_arr[zero_positions-1] += time_arr[zero_positions]
        # Create new region and time arrays that exclude the zero positions
        region_arr_new, time_arr_new = region_arr[~mask], time_arr[~mask]
        # Extend the new region and time arrays with zeros to match the original length
        region_arr_new, time_arr_new = np.concatenate([region_arr_new, np.zeros(len(
            zero_positions))]), np.concatenate([time_arr_new, np.zeros(len(zero_positions))])
        # Replace the original region and time arrays in the row with the new arrays
        row[region_cols], row[time_cols] = region_arr_new, time_arr_new
        return row

    patients = patients.apply(delete_OTF, axis=1)
    patients = patients.fillna(0)
    patients[region_cols+['isAdmitted']
             ] = patients[region_cols+['isAdmitted']].astype(int)
    patients[time_cols] = patients[time_cols].astype(float)
    patients = patients.iloc[:, 1:]
    patients.drop_duplicates(keep='first', inplace=True)
    return patients


def evaluate_roc_model(predict_proba, labels):
    """Evaluate the model using ROC curve

    Parameters
    ----------
    predict_proba: the predicted probabilities of the positive class
    labels: the labels

    returns
    -------
    roc_auc: the area under the ROC curve
    fpr: false positive rate
    tpr: true positive rate
    threshold: the threshold used to calculate the ROC curve
    """
    fpr, tpr, threshold = roc_curve(labels, predict_proba)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr, threshold


def draw_roc_curve(predict_proba, labels):
    """Draw ROC curve for the model

    Parameters
    ----------
    predict_proba: the predicted probabilities of the positive class
    labels: the labels
    """
    roc_auc, fpr, tpr, threshold = evaluate_roc_model(
        predict_proba, labels)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
