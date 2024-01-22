from __future__ import annotations
import numpy as np
from . import structure_mask as sm

class EffectFunction:
    def __init__(
            self,
            dose_array,
            struct_mask,
            shrink_pos: int=None,
            shrink_margin: int=None
    ):
        self.dose_array = dose_array
        self.struct_mask = struct_mask
        self.volume = np.sum(self.struct_mask)
    

    

class Serial(EffectFunction):
    def __init__(self,dose_array,struct_mask,exponent):
        super().__init__(dose_array,struct_mask)
        self.exponent = exponent
    
    def calculate(self):
        dose_pow = np.power(self.dose_array,self.exponent)
        total_dose = np.sum(np.multiply(dose_pow,self.struct_mask))
        dose_vol = total_dose/self.volume
        d_eff = np.power(dose_vol,1/self.exponent)
        return d_eff

class Parallel(EffectFunction):
    def __init__(self,dose_array,struct_mask,exponent,ref_dose):
        super().__init__(dose_array,struct_mask)
        self.exponent = exponent
        self.ref_dose = ref_dose
    
    def calculate(self):
        dose_pow = 1/(1+np.power(np.divide(self.ref_dose,self.dose_array,out=np.zeros_like(self.dose_array),where=self.dose_array!=0),self.exponent))
        total_dose = np.sum(np.multiply(dose_pow,self.struct_mask))
        v_eff=total_dose/self.volume*100
        return v_eff

class Quadratic(EffectFunction):
    def __init__(self,dose_array,struct_mask,ref_dose):
        super().__init__(dose_array,struct_mask)
        self.ref_dose = ref_dose

    def calculate(self):
        dose_diff = (self.dose_array-self.ref_dose)
        high_dose = np.multiply((dose_diff>0),np.sqrt(self.struct_mask))
        overdose = np.sum(np.power(np.multiply(dose_diff,high_dose),2))
        RMSE = np.sqrt(overdose/self.volume)
        return RMSE

class EUD(EffectFunction):
    def __init__(self,dose_array,struct_mask,alpha):
        super().__init__(dose_array,struct_mask)
        self.alpha = alpha

    def calculate(self):
        dose = np.exp(-self.alpha*self.dose_array)
        dose = np.multiply(dose,self.struct_mask)
        lnEUD = np.log(np.sum(dose)/self.volume)
        EUD = -1/self.alpha*lnEUD
        return EUD

class gEUD(EffectFunction):
    def __init__(self,dose_array,struct_mask,alpha):
        super().__init__(dose_array,struct_mask)
        self.alpha =alpha

    def calculate(self):
        dose = np.power(self.dose_array,self.alpha)
        dose = np.multiply(dose,self.struct_mask)
        gEUD = np.power(np.sum(dose/self.volume),1/self.alpha)
        return gEUD

class Mean(EffectFunction):

    def calculate(self):
        dose = np.multiply(self.dose_array,self.struct_mask)
        mean = np.sum(dose)/self.volume
        return mean

