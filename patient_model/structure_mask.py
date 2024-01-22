import numpy as np 
import cv2
import scipy
import multiprocessing
  
# Using sys.getrecursionlimit() method  
# to find the current recursion limit 

'''
    function to match the structures with the positions in the DICOM file

    Input:
            -organ: vector of all structures that are important
            -structure_file: DICOM structure file with all structure data
    
    Output:
            -nums: vector of structureposition in the structure file in the order given by organ
'''

def struct_position(organ,structure_file):
    speicher = {}
    nums = np.zeros(len(organ),dtype=int)
    for idx, struc_name in enumerate(list(structure_file.StructureSetROISequence)):
        speicher.update({struc_name.ROIName : idx})
    for idx, organ_name in enumerate(organ):
        if not(type(organ_name)==list):
            nums[idx]=speicher[organ_name]
        else:
            if organ_name[0] in speicher:
                nums[idx] = speicher[organ_name[0]]
            elif organ_name[1] in speicher:
                nums[idx] =speicher[organ_name[1]]
            elif organ_name[2] in speicher:
                nums[idx] =speicher[organ_name[2]]
            else:
                print('Structure not found in structurefile')
    return nums


'''
    Function to apply a bresham line algorithm on a structure set
    added points are append at the end of the structure sequence

    Input:
            -2 dimensional numpy array shape (x,3)
    
    Output:
            -2 dimensional numpy array shape (x+addedpoints,3)
'''


def bresham_line(struct_points):
    struct1 = np.vstack([struct_points,struct_points[0,:]])
    diff = struct1[1:,:]-struct_points[:,:]
    added_points = []
    for x,difference in enumerate(diff):
        if max(abs(difference))>1:
            for count in range(0,max(abs(difference))):
                added_points.append([int(y) for y in np.round(struct_points[x,:]+difference/max(abs(difference))*count)])
    if added_points:
        points = np.concatenate((struct_points,np.array(added_points)))
        return points
    else:
        return struct_points

'''
    create filled contours based on the DICOM structure file information
    contour information is saved in structure_grid

    Input: 
            -organposition: position of important organ in the structure file
            -ctrs: array form of all contour information
            -reference_point: origin of structure coordinate system
            -mr_size: size of mr_image
            -structure_grid: 4D array of zeros with dimensions: 144X144Xamount of mr_slicesXnumber of organs

    Output: 
            - 
'''
def flood_fill2(array,x,y, old, new):
        toFill = set()
        toFill.add((x,y))
        while not toFill == set():
            (x,y) = toFill.pop()
            if not array[x,y] == old:
                continue
            array[x,y]= new
            toFill.add((x-1,y))
            toFill.add((x+1,y))
            toFill.add((x,y-1))
            toFill.add((x,y+1))
        array=(array>0).astype(int)
        
'''def flood_fill(x ,y, old, new):
    # we need the x and y of the start position, the old value,
    # and the new value
    # the flood fill has 4 parts
    # firstly, make sure the x and y are inbounds
    if x < 0 or x >= field.shape[0] or y < 0 or y >= field.shape[1] :
        return
    # secondly, check if the current position equals the old value
    if field[x,y] != old:
        return

    # thirdly, set the current position to the new value
    field[x,y] = new
    # fourthly, attempt to fill the neighboring positions
    flood_fill(x+1, y, old, new)
    flood_fill(x-1, y, old, new)
    flood_fill(x, y+1, old, new)
    flood_fill(x, y-1, old, new)'''




def create_contours(nums, ctrs, reference_point, cum_org, structure_grid,spacing):
    for idx,org in enumerate(nums):
        slice_count=-1
        for contourslice in list(ctrs[org].ContourSequence):
            date = np.reshape(np.array(contourslice.ContourData),(-1,3))
            date = date-reference_point
            date = np.round(date/[spacing,spacing,1]) 
            date = date.astype(int)
            if not date[0,2] == slice_count and not slice_count==-1:
                input= (structure_grid[:,:,slice_count,cum_org[idx]]).copy()
                cv2.floodFill(input,None,(0,0),2,flags=4)
                structure_grid[:,:,slice_count,cum_org[idx]] = input<2
            # use polygon function to fill contour build with structure information
            date = bresham_line(date)
            structure_grid[date[:,0],date[:,1],date[0,2],cum_org[idx]]=1
            slice_count=date[0,2]
        input= (structure_grid[:,:,slice_count,cum_org[idx]]).copy()
        cv2.floodFill(input,None,(0,0),2,flags=4)
        structure_grid[:,:,slice_count,cum_org[idx]] = input<2
            

def create_voi(struct,voi,dose_ref,struct_ref,spacing):
    for x in range(0,voi.shape[0]):
        xnew = (x+dose_ref[0]-struct_ref[0])/spacing
        if xnew<0 or xnew>struct.shape[0]-1:
           continue 
        intx = int(xnew)
        diffx = (xnew-intx)/spacing
        for y in range(0,voi.shape[1]):
            ynew = (y+dose_ref[1]-struct_ref[1])/spacing
            if ynew<0 or ynew>struct.shape[1]-1:
                continue
            inty = int(ynew)
            diffy = (ynew-inty)/spacing
            for z in range(0,voi.shape[2]):
                znew = (z+dose_ref[2]-struct_ref[2])
                if znew<0 or znew>struct.shape[2]-1:
                    continue
                intz = int(znew)
                diffz = znew-intz
                voi[x,y,z]=struct[intx:intx+2,inty:inty+2,intz:intz+2]@[diffz,1-diffz]@[diffy,1-diffy]@[diffx,1-diffx]

def create_voi2(struct,voi,dose_ref,struct_ref,spacing):
    coordx =np.linspace(0,voi.shape[0]-1,voi.shape[0], dtype=int)
    coordy =np.linspace(0,voi.shape[1]-1,voi.shape[1], dtype=int)
    coordz =np.linspace(0,voi.shape[2]-1,voi.shape[2], dtype=int)
    coord1= (coordx+dose_ref[0]-struct_ref[0])/spacing
    coord2= (coordy+dose_ref[1]-struct_ref[1])/spacing
    coord3= (coordz+dose_ref[2]-struct_ref[2])
    intcoord1= coord1.astype(int)
    intcoord2= coord2.astype(int)
    intcoord3= coord3.astype(int)
    diffx = coord1-intcoord1
    diffy = coord2-intcoord2
    diffz =  coord3-intcoord3
    for x in coordx:
        if intcoord1[x]<0 or intcoord1[x]>struct.shape[0]-2:
           continue 
        for y in coordy:
            if intcoord2[y]<0 or intcoord2[y]>struct.shape[1]-2:
                continue
            for z in coordz:
                if intcoord3[z]<0 or intcoord3[z]>struct.shape[2]-2:
                    continue
                voi[x,y,z]=struct[intcoord1[x]:intcoord1[x]+2,intcoord2[y]:intcoord2[y]+2,intcoord3[z]:intcoord3[z]+2]@[diffz[z],1-diffz[z]]@[diffy[y],1-diffy[y]]@[diffx[x],1-diffx[x]]

def create_voi3(struct,voi,dose_ref,struct_ref,spacing):
    coordx =np.linspace(0,voi.shape[0]-1,voi.shape[0], dtype=int)
    coordy =np.linspace(0,voi.shape[1]-1,voi.shape[1], dtype=int)
    coordz =np.linspace(0,voi.shape[2]-1,voi.shape[2], dtype=int)
    coord1= (coordx+dose_ref[0]-struct_ref[0])/spacing
    coord2= (coordy+dose_ref[1]-struct_ref[1])/spacing
    coord3= (coordz+dose_ref[2]-struct_ref[2])
    intcoord1= coord1.astype(int)
    intcoord2= coord2.astype(int)
    intcoord3= coord3.astype(int)
    diffx = ((coord1-intcoord1)/spacing).reshape((voi.shape[0],1,1))
    diffy = ((coord2-intcoord2)/spacing).reshape((1,voi.shape[1],1))
    diffz = ((coord3-intcoord3)/spacing).reshape((1,1,voi.shape[2]))
    xlow = np.max([0,intcoord1[0]])
    xhigh = np.min([struct.shape[0]-2,intcoord1[-1]])
    ylow = np.max([0,intcoord2[0]])
    yhigh = np.min([struct.shape[1]-2,intcoord2[-1]])
    zlow = np.max([0,intcoord3[0]])
    zhigh = np.min([struct.shape[2]-2,intcoord3[-1]])
    #for x in range(xlow,xhigh+1):
    voi[coordx[xlow]:coordx[xhigh],coordy[ylow]:coordy[yhigh],coordz[zlow]:coordz[zhigh]] = np.multiply(struct[xlow:xhigh,ylow:yhigh,zlow:zhigh],diffx[xlow:xhigh])+np.multiply(struct[xlow+1:xhigh+1,ylow:yhigh,zlow:zhigh],1-diffx[xlow:xhigh])
    #for y in range(ylow,yhigh+1):
    voi[coordx[xlow]:coordx[xhigh],coordy[ylow]:coordy[yhigh],coordz[zlow]:coordz[zhigh]] = np.multiply(struct[xlow:xhigh,ylow:yhigh,zlow:zhigh],diffy[ylow:yhigh])+np.multiply(struct[xlow:xhigh,ylow+1:yhigh+1,zlow:zhigh],1-diffy[ylow:yhigh])
    #for z in range(zlow,zhigh+1):
    voi[coordx[xlow]:coordx[xhigh],coordy[ylow]:coordy[yhigh],coordz[zlow]:coordz[zhigh]] = np.multiply(struct[xlow:xhigh,ylow:yhigh,zlow:zhigh],diffz[zlow:zhigh])+np.multiply(struct[xlow:xhigh,ylow:yhigh,zlow+1:zhigh+1],1-diffz[zlow:zhigh])


def create_voi4(struct, voi, dose_ref, struct_ref, spacing):
    shape_voi = voi.shape
    shape_struct = struct.shape

    coordx = np.arange(shape_voi[0])
    coordy = np.arange(shape_voi[1])
    coordz = np.arange(shape_voi[2])

    dose_offset = np.array(dose_ref) - np.array(struct_ref)
    dose_offset /= spacing

    indices_x = (coordx + dose_offset[0]).astype(int)
    indices_y = (coordy + dose_offset[1]).astype(int)
    indices_z = (coordz + dose_offset[2]).astype(int)

    diff_x = (coordx - indices_x) / spacing
    diff_y = (coordy - indices_y) / spacing
    diff_z = (coordz - indices_z)

    for x in coordx:
        if indices_x[x] < 0 or indices_x[x] > shape_struct[0] - 2:
            continue
        for y in coordy:
            if indices_y[y] < 0 or indices_y[y] > shape_struct[1] - 2:
                continue
            for z in coordz:
                if indices_z[z] < 0 or indices_z[z] > shape_struct[2] - 2:
                    continue
                voi[x, y, z] = np.sum(
                    struct[indices_x[x]:indices_x[x] + 2, indices_y[y]:indices_y[y] + 2,
                           indices_z[z]:indices_z[z] + 2] *
                    [diff_z[z], 1 - diff_z[z]] *
                    [diff_y[y], 1 - diff_y[y]] *
                    [diff_x[x], 1 - diff_x[x]]
                )

'''def inner_loop(x,struct, voi, spacing,dose_offset,shape_struct,shape_voi):
    x_new = (x + dose_offset[0])/spacing
    if not (0 <= x_new < shape_struct[0] - 1):
        return
    int_x = int(x_new)
    diff_x = x_new - int_x
    for y in range(shape_voi[1]):
        y_new = (y + dose_offset[1])/spacing
        if not (0 <= y_new < shape_struct[1] - 1):
            continue
        int_y = int(y_new)
        diff_y = y_new - int_y

        for z in range(shape_voi[2]):
            z_new = z + dose_offset[2]
            if not (0 <= z_new < shape_struct[2] - 1):
                continue
            int_z = int(z_new)
            diff_z = z_new - int_z
            int_x2 = int_x + 1
            int_y2 = int_y + 1
            int_z2 = int_z + 1
            c000 = struct[int_x, int_y, int_z]
            c001 = struct[int_x, int_y, int_z2]
            c010 = struct[int_x, int_y2, int_z]
            c011 = struct[int_x, int_y2, int_z2]
            c100 = struct[int_x2, int_y, int_z]
            c101 = struct[int_x2, int_y, int_z2]
            c110 = struct[int_x2, int_y2, int_z]
            c111 = struct[int_x2, int_y2, int_z2]


            interpolated_value = (
                c111 * (1 - diff_x) * (1 - diff_y) * (1 - diff_z) +
                c110 * (1 - diff_x) * (1 - diff_y) * diff_z +
                c101 * (1 - diff_x) * diff_y * (1 - diff_z) +
                c100 * (1 - diff_x) * diff_y * diff_z+
                c011 * diff_x * (1 - diff_y) * (1 - diff_z) +
                c010 * diff_x * (1 - diff_y) * diff_z +
                c001 * diff_x * diff_y * (1 - diff_z) +
                c000 * diff_x * diff_y * diff_z
            )

            voi[x, y, z] = interpolated_value

def create_test(struct, voi, dose_ref, struct_ref, spacing):
    shape_voi = voi.shape
    shape_struct = struct.shape

    dose_offset = np.array(dose_ref) - np.array(struct_ref)
    #inv_spacing = 1.0 / spacing
    for x in range(shape_voi[0]):
        process = multiprocessing.Process(target=inner_loop, args = (struct,voi,dose_offset,shape_struct,shape_voi))
        process.start()
    pool_obj = multiprocessing.Pool()
    pool_obj.map(inner_loop)'''
        

def create_voi5(struct, voi, dose_ref, struct_ref, spacing):
    shape_voi = voi.shape
    shape_struct = struct.shape
    dose_offset = (np.array(dose_ref) - np.array(struct_ref))/spacing
    coord1= (np.arange(0,voi.shape[0])+dose_ref[0]-struct_ref[0])/spacing
    coord2= (np.arange(0,voi.shape[1])+dose_ref[1]-struct_ref[1])/spacing
    coord3= (np.arange(0,voi.shape[2])+dose_ref[2]-struct_ref[2])
    low1 = max([coord1[0],0])
    low2 = max([coord2[0],0])
    low3 = max([coord3[0],0])
    high1 = min([coord1[-1],shape_struct[0]-2])
    high2 = min([coord2[-1],shape_struct[1]-2])
    high3 = min([coord3[-1],shape_struct[2]-2])

    for x in coord1[low1:high1]:
        int_x = int(x)
        fract_x = x - int_x
        for y in coord2[low2:high2]:
            int_y = int(y)
            fract_y = y - int_y
            for z in coord3[low3:high3]:
                int_z = int(z)
                fract_z = z - int_z

                c000 = struct[int_x, int_y, int_z]
                c001 = struct[int_x, int_y, int_z + 1]
                c010 = struct[int_x, int_y + 1, int_z]
                c011 = struct[int_x, int_y + 1, int_z + 1]
                c100 = struct[int_x + 1, int_y, int_z]
                c101 = struct[int_x + 1, int_y, int_z + 1]
                c110 = struct[int_x + 1, int_y + 1, int_z]
                c111 = struct[int_x + 1, int_y + 1, int_z + 1]

                voi[x, y, z] = (
                    c111 * (1 - fract_x) * (1 - fract_y) * (1 - fract_z) +
                    c110 * (1 - fract_x) * (1 - fract_y) * fract_z +
                    c101 * (1 - fract_x) * fract_y * (1 - fract_z) +
                    c100 * (1 - fract_x) * fract_y * fract_z +
                    c011 * fract_x * (1 - fract_y) * (1 - fract_z) +
                    c010 * fract_x * (1 - fract_y) * fract_z +
                    c001 * fract_x * fract_y * (1 - fract_z) +
                    c000 * fract_x * fract_y * fract_z
                )


def create_voi6(struct, voi, dose_ref, struct_ref, spacing):
    shape_voi = voi.shape
    shape_struct = struct.shape

    dose_offset = np.array(dose_ref) - np.array(struct_ref)
    #inv_spacing = 1.0 / spacing

    for x in range(shape_voi[0]):
        x_new = (x + dose_offset[0])/spacing
        if not (0 <= x_new < shape_struct[0] - 1):
            continue
        int_x = int(x_new)
        if not(np.any(struct[int_x,:,:]>0) or np.any(struct[int_x+1,:,:]>0)):
            continue
        diff_x = x_new - int_x
        for y in range(shape_voi[1]):
            y_new = (y + dose_offset[1])/spacing
            if not (0 <= y_new < shape_struct[1] - 1):
                continue
            int_y = int(y_new)
            if not(np.any(struct[int_x,int_y:int_y+2,:]>0) or np.any(struct[int_x+1,int_y:int_y+2,:]>0)):
                continue
            diff_y = y_new - int_y

            for z in range(shape_voi[2]):
                z_new = z + dose_offset[2]
                if not (0 <= z_new < shape_struct[2] - 1):
                    continue
                int_z = int(z_new)
                diff_z = z_new - int_z

                int_x2 = int_x + 1
                int_y2 = int_y + 1
                int_z2 = int_z + 1

                c000 = struct[int_x, int_y, int_z]
                c001 = struct[int_x, int_y, int_z2]
                c010 = struct[int_x, int_y2, int_z]
                c011 = struct[int_x, int_y2, int_z2]
                c100 = struct[int_x2, int_y, int_z]
                c101 = struct[int_x2, int_y, int_z2]
                c110 = struct[int_x2, int_y2, int_z]
                c111 = struct[int_x2, int_y2, int_z2]


                interpolated_value = (
                    c111 * (1 - diff_x) * (1 - diff_y) * (1 - diff_z) +
                    c110 * (1 - diff_x) * (1 - diff_y) * diff_z +
                    c101 * (1 - diff_x) * diff_y * (1 - diff_z) +
                    c100 * (1 - diff_x) * diff_y * diff_z+
                    c011 * diff_x * (1 - diff_y) * (1 - diff_z) +
                    c010 * diff_x * (1 - diff_y) * diff_z +
                    c001 * diff_x * diff_y * (1 - diff_z) +
                    c000 * diff_x * diff_y * diff_z
                )

                voi[x, y, z] = interpolated_value
'''test = np.array([[1,2,3],[10,6,3],[3,9,3]])
con_data = bresham_line(test)
field = np.ones((15,15))


field[con_data[:,0],con_data[:,1]]=2
print(field)
flood_fill(0, 0, 1, 0)
field = (field>0).astype(int)'''

def create_shrink_kernel(shrink,grid_size):
    shrink_norm = shrink/grid_size
    sw = np.ceil(shrink_norm)
    x = np.arange(-sw,sw+1,1)
    x,y,z = np.meshgrid(x,x,x)
    kernel = x**2+y**2+z**2
    ball=kernel<=shrink_norm**2
    return ball


def create_shrink_voi(vois,pos_name,pos_shrink_target,shrink_margin,grid_size):
    cf_voi = vois[pos_name,:,:,:]
    for idx,position in enumerate(pos_shrink_target):
        if shrink_margin[idx]>0:
            target_voi = vois[position,:,:,:]
            shrink_kernel = create_shrink_kernel(shrink_margin[idx],grid_size)
            conv = scipy.ndimage.convolve(target_voi,shrink_kernel)
            cf_voi = cf_voi-(conv>0)
        else:
            cf_voi = cf_voi-vois[position,:,:,:]
    cf_voi[cf_voi<0.05]=0
    return cf_voi