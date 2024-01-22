import patient_model as pat
import os, os.path
import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import block_reduce
import sys
from file_manager import log_folder_data
import yaml

#organ = [["PTV_60"],["PTV_57_6"],["Anorectum"],["Bladder","Bladder_1"],["patient"],["Bone_Pelvic_L","Pelvis_L","Pelvis_L_1"],\
#         ["Bone_Pelvic_R","Pelvis_R","Pelvis_R_1"],["Sacrum","Sacrum_1"],["FemoralHead_L","Femur_L","Femur_L_1"],["FemoralHead_R","Femur_R","Femur_R_1"]]
#cum_org = [0,1,2,3,4,5,5,6,7,7]
#prescribed_dose=[60,57.6]

def main(datadir,yamlfile):
    log = {}
    log_folder_data(datadir,log)
    if "RTStruct" in log:
        struc_file = dicom.read_file(datadir+str(log["RTStruct"][0]),force=True)
    else:
        print("No Structure Set found. Aborting...")
    if "RTDose" in log:
        dose_file = dicom.read_file(datadir+str(log["RTDose"][0]),force=True)
        dose_file.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
    else:
        print("No Structure Set found. Aborting...")
    if "CTImage" in log:
        image_file = dicom.read_file(datadir+str(log["CTImage"][0]),force=True)
        image_file.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
    elif "MRImage" in log:
        image_file = dicom.read_file(datadir+str(log["MRImage"][0]),force=True)
    else:
        print("No Structure Set found. Aborting...")

    #read yaml file and extract target and OARs used in costfunctions

    with open(yamlfile,'r') as file:
        costfunctions = yaml.safe_load(file)

    organ=list(costfunctions["organs"].keys())
    cum_org = []
    for x in range(0,len(costfunctions["organs"].keys())):
        cum_org.append(int(list(costfunctions["organs"].values())[x]["organ_num"]))
    #prescribed_dose = list(costfunctions["prescribed_dose"]["dose"])

    spacing = image_file.PixelSpacing[0]
    #create zero arrays with calculated dimensions
    structure_grid2 = np.zeros((image_file.pixel_array.shape[0],image_file.pixel_array.shape[1],len(log["MRImage"]),max(cum_org)+1), dtype=np.uint8)

    #calculate originpoints of coordinatesystems of structure and dose files
    #dose_point = np.array(dose_file.ImagePositionPatient)-1.5
    dose_point = np.array(dose_file.ImagePositionPatient)
    dose_point_large = dose_point-[1,1,1]


    #reference point to arrange structureinformation
    '''reference_point = np.array(image_file.ImagePositionPatient)-(image_file.PixelSpacing[0]/2)
    reference_point[2] = -int(len(log["MRImage"])*image_file.SpacingBetweenSlices/2)
    print(reference_point)'''
    reference_point = np.array(image_file.ImagePositionPatient)
    reference_point[2] = -int(len(log["MRImage"])*image_file.SpacingBetweenSlices/2)+0.5

    #contour information in array
    ctrs=struc_file.ROIContourSequence

    dose_arr= np.array(dose_file.pixel_array)*float(dose_file.DoseGridScaling)
    dose_arr = np.transpose(dose_arr)

    print('matching organs to structure file position...')
    nums = pat.struct_position(organ,struc_file)

    #creates contours
    print('creating filled contoures based on structure informations...')
    pat.create_contours(nums, ctrs, reference_point, cum_org, structure_grid2,spacing)

    trans_dose_file2 = np.transpose(np.zeros(tuple(s*3 for s in dose_file.pixel_array.shape)+(structure_grid2.shape[3],)))


    print("Calculate Voi Occupancy in dose grid coordinates...")
    vois =np.empty([structure_grid2.shape[3],dose_file.pixel_array.shape[2],dose_file.pixel_array.shape[1],dose_file.pixel_array.shape[0]])
    for i in range(0, structure_grid2.shape[3]):
        pat.create_voi6(structure_grid2[:,:,:,i],trans_dose_file2[i,:,:,:],dose_point_large,reference_point,spacing) 
        vois[i,:,:,:] = block_reduce(trans_dose_file2[i,:,:,:], block_size=(3,3,3), func=np.mean, cval=np.mean(trans_dose_file2))

    print("Calculate Isoeffects and shrinking Costfunction Occupancy...")
    #print(list(costfunctions["organs"].values())[2]["costfunctions"].values())
    for idx,organval in enumerate(list(costfunctions["organs"].values())):
        for idy,cost in enumerate(organval["costfunctions"].values()):
            if "shrink" in cost:
               cost_arr = pat.create_shrink_voi(vois,cum_org[idx],list(cost["shrink"]),list(cost["shrink_margin"]),3) 
            else:
                cost_arr = vois[cum_org[idx],:,:,:]
                cost_arr[cost_arr<0.05]=0
            
            if cost["type"]=="Serial":
                result =pat.Serial(dose_arr,cost_arr,cost["exponent"])
            elif cost["type"]=="Parallel":
                result = pat.Parallel(dose_arr,cost_arr,cost["exponent"],cost["dose"])
            elif cost["type"]=="Quadratic":
                result = pat.Quadratic(dose_arr,cost_arr,cost["dose"])
            elif cost["type"]=="EUD":
                result =pat.EUD(dose_arr,cost_arr,cost["alpha"])
            elif cost["type"]=="gEUD":
                result =pat.gEUD(dose_arr,cost_arr,cost["alpha"])
            else:
                print("type not found")
                continue
            print(organ[idx]," ",cost["type"],": ",result.calculate())
            


main(sys.argv[1],sys.argv[2])

    
    #calculate structure and dose file dimensions
