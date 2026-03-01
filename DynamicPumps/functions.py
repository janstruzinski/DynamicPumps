from scipy import integrate

def get_dP_from_H(fluid, H, p_0, T_0):
    """A function to get pressure rise from head"""
    dP = ...
    return dP

def get_H_from_dp(fluid, pressure_rise, p_0, T_0):
    """A function to get head from pressure rise"""
    # Integrate dp / (density * g) from p_0 to p_1 = p_0 + pressure_rise to get head
    H = integrate.quad(lambda p: 1 / fluid.get_density(p, T_0), p_0, p_0 + pressure_rise)[1] / 9.80665
    return H