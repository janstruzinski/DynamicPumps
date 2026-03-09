from scipy import integrate
from scipy.optimize import toms748

def get_dP_from_H(fluid, H, p_0, T_0):
    """A function to get pressure rise from head"""
    # TODO Update docstring

    # First create a residual function to be solved
    def residual(dp):
        return get_H_from_dp(fluid, dp, p_0, T_0) - H
    # Get first estimate of dp
    rho = fluid.get_density(p_0, T_0)
    dp_estimate = rho * H * 9.80665
    # Solve for dp
    dp = toms748(f=residual, a=0, b=2*dp_estimate)[0]
    return dp

def get_H_from_dp(fluid, pressure_rise, p_0, T_0):
    """A function to get head from pressure rise"""
    #TODO Update docstring

    # Integrate dp / (density * g) from p_0 to p_1 = p_0 + pressure_rise to get head
    H = integrate.quad(lambda p: 1 / fluid.get_density(p, T_0), p_0, p_0 + pressure_rise)[1] / 9.80665
    return H