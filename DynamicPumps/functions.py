from scipy import integrate
from scipy.optimize import toms748

def get_dP_from_H(fluid, H, p_0, T_0):
    """A function to get pressure rise from head.

    :param fluid: Fluid object
    :param float or int H: Pump head, m.
    :param float or int p_0: Initial pressure with respect to which pump head is given, Pa.
    :param float or int T_0: Flow temperature, K.

    :return: Pressure difference from pump head, Pa.
    :rtype: float
    """

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
    """A function to get head from pressure rise

    :param fluid: Fluid object
    :param float or int pressure_rise: Pump pressure rise, Pa.
    :param float or int p_0: Initial pressure with respect to which pump head is given, Pa.
    :param float or int T_0: Flow temperature, K.

    :return: Pump head, m.
    :rtype: float
    """

    # Integrate dp / (density * g) from p_0 to p_1 = p_0 + pressure_rise to get head
    H = integrate.quad(lambda p: 1 / fluid.get_density(p, T_0), p_0, p_0 + pressure_rise)[1] / 9.80665
    return H