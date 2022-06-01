from cc3d.core.PySteppables import *
from random import uniform
import numpy as np

phi0 = {{phi0}}
m_current = {{m_current}}
tau= {{tau}}
half_cone= {{half_cone}}
depth_sensed = 1 # this will be tied to the alpha field, useful for glutaraldehyde. 
# If the alpha is sensed at the bottom of collagen, then this is 1, thus increasing the cell forces
# USE ALPHA_THRESH = 0.0001 FOR DEPTH SENSED at the bottom or not

class UpdatePhi(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self,frequency)

    def start(self):
        self.shared_steppable_vars['cluster_theta']= 0.0

    def step(self, mcs):

        count_boundary= 0
        for cell in self.cell_list_by_type(self.CELL):
            cell.dict['IsBoundary'] = 0
            for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                if not neighbor:
                    cell.dict['IsBoundary'] = 1
                    count_boundary+= 1
                    break

        pop_com = np.array([0.0,0.0])
        count= 0
        for cell in self.cell_list_by_type(self.CELL):
            if cell.dict['IsBoundary']== 0:
                pop_com += np.array([cell.xCOM,cell.yCOM])
                count+= 1
        pop_com /= count
        self.shared_steppable_vars['pop_com'] = pop_com



        field_phi = self.field.phi_field
        field_rm_abi = self.field.rm_abi
        field_alpha = self.field.alpha
        field_phi[:,:,:] = 0.0
        field_rm_abi[:,:,:] = 0.0

        field_IsBoundary = self.field.IsBoundary
        field_IsBoundary[:,:,:] = 0.0

        

        if mcs%tau == 0:
            cluster_theta = (np.random.randint(0,23))*360/24*(3.14/180) # THIS IS THE RANDOM DIRECTION PICKED
            self.shared_steppable_vars['cluster_theta'] = cluster_theta

#        cluster_theta = (mcs//tau)*60*3.14/180 # np.random.randint(0,23) # THIS IS THE RANDOM DIRECTION PICKED
        cluster_theta= self.shared_steppable_vars['cluster_theta']

        cluster_theta = self.shared_steppable_vars['cluster_theta']
        cell_angle_list = []
        alpha_list= []
        # AR= self.shared_steppable_vars('AR')
        count_active= 0
        
        for cell in self.cell_list_by_type(self.CELL):
            phi = phi0
            rm_abi = 0
            # cell.dict['phi'] = phi
            cell.dict['rm_abi'] = rm_abi
            IsBoundary = cell.dict['IsBoundary']
            if IsBoundary == 1:
                rm_abi = 0.1 # ADJUST IF TOO LOW
                cell_com = np.array([cell.xCOM, cell.yCOM])
                cell_angle = FindAngle(cell_com, pop_com)
                cell_angle_list.append(cell_angle)
                alpha_cell= field_alpha[cell.xCOM, cell.yCOM, cell.zCOM]
                alpha_list.append(alpha_cell)
                if (np.absolute(cell_angle-cluster_theta) < half_cone*3.14/180 ) or (np.absolute(2*3.14-(cell_angle-cluster_theta)) < half_cone*3.14/180 ):
                    rm_abi= 1
                    phi= (mcs >= 50)*(m_current == 110)*(depth_sensed) + phi0# 1/2
                    count_active+= 1
                else:
                    rm_abi=0.1

                cell.dict['rm_abi'] = rm_abi
                cell.dict['phi'] = phi

            pixel_list = self.get_cell_pixel_list(cell)
            for pixel_tracker_data in pixel_list:
                px = pixel_tracker_data.pixel
                field_phi[px.x,px.y,px.z] = phi
                field_rm_abi[px.x,px.y,px.z] = rm_abi
                field_IsBoundary[px.x,px.y,px.z] = IsBoundary
        
        f_active= (1 - 2.71**(count_active/count_boundary))/(1 + 2.71**(count_active/count_boundary))
        
        for cell in self.cell_list_by_type(self.CELL):# SET PHI FOR NON REMODELING CELLS
            if cell.dict['rm_abi'] != 1:
                phi= (f_active)*(mcs >= 50)*(m_current == 110)*(depth_sensed) + phi0# 1/2
                cell.dict['phi']= phi
            pixel_list = self.get_cell_pixel_list(cell)
            for pixel_tracker_data in pixel_list:
                px = pixel_tracker_data.pixel
                field_phi[px.x,px.y,px.z] = phi

        self.shared_steppable_vars['cell_angle_list'] = cell_angle_list
        self.shared_steppable_vars['alpha_list'] = alpha_list

class DoRemodeling(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self,frequency)

    def step(self, mcs):

        field_alpha = self.field.alpha
        field_cell = self.cell_field
        pop_com = self.shared_steppable_vars['pop_com']
        cell_angle_list = self.shared_steppable_vars['cell_angle_list']
        alpha_list = self.shared_steppable_vars['alpha_list']
        
        for x, y, z in self.every_pixel():
            if not field_cell[x,y,z]: # ONLY ECM PIXELS
                pix = np.array([x,y])
                angle = FindAngle(pix,pop_com)
                alpha_propagate = FindAlpha(angle, cell_angle_list, alpha_list)
                # cluster_num = FindCluster(pix, pop_com)
                # alpha_LE = cluster_alpha[cluster_num]
                dist = np.linalg.norm(pix-pop_com)#POP_COM OR COM OF THE CELLS IN THAT GROUP
                if alpha_propagate/(dist**0.5) > field_alpha[x,y,z]:
                    field_alpha[x,y,z] = alpha_propagate/(dist**0.5) # THIS CANT BE CLUSTER PHI, IT HAS TO BE ALPHA_AVG/R
                
def FindAngle(pix, pop_com): #BOTH NP ARRAYS
    vec_to_pix = pix - pop_com
    angle_raw = np.arctan2(vec_to_pix[1], vec_to_pix[0])
    angle_raw = angle_raw + (angle_raw < 0)*(2*3.14) # CONVERT TO 0-2PI
    
    return angle_raw

def FindAlpha(angle360, cell_angle_list, alpha_list):

    temp = np.absolute(cell_angle_list-angle360)
    cell_location = np.argmin(temp)
    alpha = alpha_list[cell_location]   
    
    return alpha
            



class FindECMpolarity(SteppableBasePy): #FIND THE NET DIRECTION OF ECM POLARITY, 
# DO THE RANDOM DISTRIBUTIONS AND PICK A NEW FRONT/BACK THETA, TO BE FED INTO UPDATE_RM
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self,frequency)

    def start(self):
        #STUFF

    def step(self, mcs):       
        #STUFF

class SetCellForces(SteppableBasePy):

    def __init__(self, frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        """
        Called before MCS=0 while building the initial simulation
        """

    def step(self, mcs):

        half_theta = self.shared_steppable_vars['cluster_theta']
        for cell in self.cell_list_by_type(self.CELL):
            cell.lambdaVecX = -10*cell.dict['phi']*np.cos(half_theta)  # force component pointing along X axis - towards positive X's
            cell.lambdaVecY = -10*cell.dict['phi']*np.sin(half_theta)  # force component pointing along Y axis - towards negative Y's
            cell.lambdaVecZ = 0.0  # force component pointing along Z axis
            
            

    def finish(self):
        """
        Called after the last MCS to wrap up the simulation
        """

    def on_stop(self):
        """
        Called if the simulation is stopped before the last MCS
        """
