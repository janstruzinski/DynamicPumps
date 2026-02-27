from scipy import integrate

def get_dP_from_H(fluid, H, p_0, T_0):
    """A function to get pressure rise from head"""
    dP = ...
    return dP

def get_H_from_dP(fluid, pressure_rise, P_0, T_0):
    """A function to get head from pressure rise"""
    # Integrate dP / (density * g) from P_0 to P_1 = P_0 + pressure_rise to get head
    H = integrate.quad(lambda P: 1 / fluid.get_density(P, T_0), P_0, P_0 + pressure_rise)[1] / 9.80665
    return H