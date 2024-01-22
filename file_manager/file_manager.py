import os, os.path
import pydicom as dicom


def log_folder_data(datadir,folder_log):
    directory_data=os.listdir(datadir)
    for idx,files in enumerate(os.listdir(datadir)):
        try:
            file = dicom.dcmread(datadir+files,force=True,stop_before_pixels=True)
            file_class = file.SOPClassUID
        except:
            continue
        if file_class =="1.2.840.10008.5.1.4.1.1.481.3":
            if not "RTStruct" in folder_log:
                folder_log["RTStruct"]=[files]
            else:
                folder_log["RTStruct"].append(files)
        elif file_class =="1.2.840.10008.5.1.4.1.1.481.2":
            if not "RTDose" in folder_log:
                folder_log["RTDose"]=[files]
            else:
                folder_log["RTDose"].append(files)
        elif file_class =="1.2.840.10008.5.1.4.1.1.2":
            if not "CTImage" in folder_log:
                folder_log["CTImage"]=[files]
            else:
                folder_log["CTImage"].append(files)
        elif file_class =="1.2.840.10008.5.1.4.1.1.4":
            if not "MRImage" in folder_log:
                folder_log["MRImage"]=[files]
            else:
                folder_log["MRImage"].append(files)
        elif file_class =="1.2.840.10008.5.1.4.1.1.481.5":
            if not "RTPlan" in folder_log:
                folder_log["RTPlan"]=[files]
            else:
                folder_log["RTPlan"].append(files)
        else:
            print("unkown dicom SOPClass UID: ",file_class)
