from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
import pandas as pd
import numpy as np
from sklearn.calibration import label_binarize
from sklearn.model_selection import train_test_split


def data_preprocessing(file_name, with_id=False, first_time_preprocess=True):
    """Preprocess the data

    Parameters
    ----------
    file_name: the name of the file containing the data
    """
    # Load the data
    data = pd.read_csv(file_name, low_memory=False)
    data = data.dropna()
    data['StartTime'] = pd.to_datetime(
        data['StartTime'], format='%d/%m/%Y %H:%M')
    data['EndTime'] = pd.to_datetime(data['EndTime'], format='%d/%m/%Y %H:%M')
    data['Pathway'] = (data['Pathway'] != 'Walkin').astype('int')
    max_sequence = max(data.groupby('SpellID_Anon').size())
    area_time_sequence_head = [item for sublist in [
        [x, 'Time'+str(x)] for x in range(1, max_sequence + 1)] for item in sublist]
    col = ['SpellID_Anon', 'Pathway', 'FirstTimetoED'] + \
        area_time_sequence_head+['isAdmitted']

    patients = pd.DataFrame(columns=col)

    # encode the first time to ED
    patients[['SpellID_Anon', 'Pathway', 'FirstTimetoED']
             ] = data[data['AreaSequence'] == 1][['SpellID_Anon', 'Pathway', 'StartTime']]
    if first_time_preprocess:
        patients['FirstTimetoED'] = patients['FirstTimetoED'].apply(
            lambda x: x.hour+x.minute/60)

    # encode the admitted status
    transfer_status_list = pd.Series(data['TransferStatus'].unique())
    uch_list = transfer_status_list[~transfer_status_list.isin(
        ['Arrival-Discharge', 'Arrival', 'Discharged', 'Transfer'])]
    patient_admitted_list = data[data['TransferStatus'].isin(uch_list)]
    patients['isAdmitted'] = patients['SpellID_Anon'].isin(
        patient_admitted_list['SpellID_Anon']).astype(int)

    dict_area = pd.Series(data['CurrentArea'].unique()).to_dict()
    area_list = list(dict_area.values())
    data_grouped = data.groupby('SpellID_Anon')[
        ['AreaSequence', 'CurrentArea', 'TotalMins']].agg(list)
    data_dict = data_grouped.to_dict('index')

    def add_row(row):
        col_list = data_dict[row['SpellID_Anon']]['AreaSequence']
        patient = data_dict[row['SpellID_Anon']]
        for i in range(0, len(col_list)):
            currentArea = patient['CurrentArea'][i]
            TotalMins = patient['TotalMins'][i]
            if TotalMins == 0:
                row[col_list[i]] = 0
                row['Time'+str(int(col_list[i]))] = 0
            else:
                row[col_list[i]] = area_list.index(currentArea)
                row['Time'+str(int(col_list[i]))] = TotalMins
        return row

    patients = patients.apply(add_row, axis=1)
    region_cols = [i for i in range(1, max_sequence + 1)]
    time_cols = ['Time' + str(i) for i in range(1, max_sequence + 1)]

    """
    This code is designed to handle rows in a DataFrame where a region sequence (represented by `region_arr`) has a value of 0,
    which is considered an "of-the-floor" (OTF) event.
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
        for zero_position in zero_positions:
            if zero_position > 0:
                time_arr[zero_position-1] += time_arr[zero_position]
            else:
                time_arr[zero_position+1] += time_arr[zero_position]
        region_arr_new, time_arr_new = region_arr[~mask], time_arr[~mask]
        for i in range(0, len(region_arr_new)):
            if i < len(region_arr_new):
                if region_arr_new[i] == region_arr_new[i-1]:
                    time_arr_new[i-1] += time_arr_new[i]
                    region_arr_new = np.delete(region_arr_new, i)
                    time_arr_new = np.delete(time_arr_new, i)
                    i -= 1
            else:
                break
        # Extend the new region and time arrays with zeros to match the original length
        region_arr_new, time_arr_new = np.pad(region_arr_new, (0, len(region_arr) - len(
            region_arr_new)), 'constant'), np.pad(time_arr_new, (0, len(time_arr) - len(
                time_arr_new)), 'constant')
        # Replace the original region and time arrays in the row with the new arrays
        row[region_cols], row[time_cols] = region_arr_new, time_arr_new
        return row

    patients = patients.apply(delete_OTF, axis=1)
    patients = patients.fillna(0)
    patients[region_cols+['isAdmitted']
             ] = patients[region_cols+['isAdmitted']].astype(int)
    patients[time_cols] = patients[time_cols].astype(float)
    patients = patients[patients[1] != 0]
    if not with_id:
        patients = patients.iloc[:, 1:]
    patients.drop_duplicates(keep='first', inplace=True)
    # patients.to_csv('patients_with_id.csv', index=False)
    return patients


def process_data(df, two_classes):
    """the detail of split the data

    Parameters
    ----------
    df: the data to be split

    returns
    -------
    splited: the splited data
    """
    splited = pd.DataFrame(columns=df.columns)

    def split_data(row):
        np_area = row.values[2:-1]
        mask = np_area == 0
        first_zero = np.where(mask)[0][0]
        new_row = np_area[:first_zero]
        splited.loc[len(splited)] = row
        final_status = row.values[-1]
        for i in range(0, int(first_zero/2-1)):
            new_row = new_row[:-2]
            new_concat_row = np.concatenate([row.values[:2], new_row])
            # pad with zeros to length of patients.columns - 1
            new_concat_row = np.pad(
                new_concat_row, (0, df.columns.shape[0] - new_concat_row.shape[0] - 1), 'constant')
            if two_classes:
                new_concat_row = np.concatenate(
                    [new_concat_row, [final_status]])
            else:
                new_concat_row = np.concatenate([new_concat_row, [2]])
            new_concat_row = pd.Series(new_concat_row, index=df.columns)
            # add new row to test_splited
            splited.loc[len(splited)] = new_concat_row

    df.apply(split_data, axis=1)
    return splited


def split_data(data, test_size=0.2, two_classes=False):
    """ Split the data into training and testing sets

    Parameters
    ----------
    data: the data to be split
    test_size: the size of the testing set

    returns
    -------
    X_train, X_test, y_train, y_test: the training and testing sets
    """
    train, test = train_test_split(data, test_size=test_size, random_state=0)
    train = process_data(train, two_classes)
    test = process_data(test, two_classes)
    X_train, X_test = train.iloc[:, :-1].values, test.iloc[:, :-1].values
    y_train, y_test = train.iloc[:, -1].values, test.iloc[:, -1].values

    return X_train, X_test, y_train, y_test


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
    labels = label_binarize(labels, classes=[0, 1, 2])
    n_classes = labels.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predict_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(
            labels.ravel(), predict_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc, fpr, tpr


def draw_roc_curve(predict_proba, labels, name):
    """Draw ROC curve for the model

    Parameters
    ----------
    predict_proba: the predicted probabilities of the positive class
    labels: the labels
    """

    plt.figure()
    if predict_proba.shape[1] == 1:
        fpr, tpr, thresholds = roc_curve(labels, predict_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (area = %0.2f)' % roc_auc)
    elif predict_proba.shape[1] == 2:
        fpr, tpr, thresholds = roc_curve(labels, predict_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC curve (area = %0.2f)' % roc_auc)
    else:
        roc_auc, fpr, tpr = evaluate_roc_model(
            predict_proba, labels)
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]))
        for i in range(predict_proba.shape[1]):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(
        f'Some extension of Receiver operating characteristic to {"Binary" if predict_proba.shape[1] == 2 else "Multi"} class ({name})')
    plt.legend(loc="lower right")
    plt.show()
