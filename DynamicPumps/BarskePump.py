import copy
import warnings
import numpy as np
import functions
import scipy.optimize as opt
import scipy.interpolate as intrp
from tabulate import tabulate
import matplotlib.pyplot as plt


class BarskePump:
    def __init__(self):
        """A class for sizing and analysis of Barske pump."""

        # Diameters
        self.D_inlet = None # inlet pipe diameter, m
        self.D_0 = None # impeller eye diameter, m
        self.D_1 = None # impeller inlet diameter, m
        self.D_2 = None # impeller outlet diameter, m
        self.D_3 = None # diffuser throat diameter, m
        self.D_4 = None # diffuser outlet diameter, m
        self.D_5 = None # hub diameter, m
        self.D_1_over_D_0 = None # ratio of D_1 to D_0, -
        self.D_5_over_D_1 = None # ratio of D_5 to D_1, -

        # Widths & lengths
        self.L_1 = None # impeller width (axial length) at the inlet, m
        self.L_2 = None # impeller width at the outlet, m
        self.L_1_over_D_1 = None # ratio of L_1 and D_1, -
        self.L_diffuser = None # diffuser length, m

        # Areas
        self.A_0 = None # impeller eye area, m^2
        self.A_3 = None # impeller throat diffuser area, m^2
        self.A_4 = None # impeller exit diffuser area, m^2
        self.A_4_over_A_3 = None # diffuser area ratio, -

        # Clearances
        self.s_ax = None # axial clearance, m
        self.s_rad = None  # radial clearance, m

        # Thicknesses
        self.t_0 = None # impeller hub thickness, m
        self.t_1 = None # impeller blade thickness at inlet, m
        self.t_2 = None # impeller blade thickness at outlet, m

        # Angles
        self.alpha_0 = None # impeller blade forward edge angle wrt. rotation axis, degrees
        self.alpha_1 = None # impeller blade backward edge angle wrt. rotation axis, degrees
        self.alpha_2 = None # sharpening angle of the radial blade wrt. tangent of D_1, degrees
        self.alpha_diffuser = None # diffuser full angle, degrees

        # Other
        self.n_blades = None # number of blades
        self.specific_speed = None # specific speed of the pump at Best Efficiency Point, EU units (m, RPM, m^3/h)
        self.eta_losses_design = None  # fraction of dynamic head lost in the diffuser used for sizing, -.
        self.K_factor_design = None # factor for prerotation at zero flow as a fraction of inlet tip speed used in sizing with
        # Lock's analysis method, -
        self.no_prerotation_design = None # boolean whether no prerotation should be assumed in case of sizing with
        # Lobanoff method.
        self.flow_coefficient_BEP = None # outlet flow coefficient at BEP, -
        self.static_head_coefficient_BEP_Barske = None # static head coefficient at BEP from Lobanoff sizing method for head calculations
        # with Barske method, -.
        self.static_head_coefficient_BEP_Lock = None # static head coefficient at BEP from Lock's method, -.

        #TODO Add forces

        # Analysis results - dictionary to store analysis inputs and results.
        self.analysis_results = {"method": None, # Method used for analysis, -
                                 "fluid": None, # Fluid object used for analysis, -
                                 "RPM": None, # Rotations Per Minute, 1 / minute
                                 "omega": None,  # Angular speed, rad / s
                                 "mdot": None, # Massflow through pump, kg / s
                                 "dp": None,  # Pressure rise across the pump, Pa
                                 "Q": None, # Volumetric flow through pump (defined with p_0)
                                 "H_total_real": None, # Real total head of the pump, m
                                 "H_static_real": None,  # Real static head of the pump, m
                                 "H_losses": None,  # Head losses of the pump, m
                                 "H_total_ideal": None,  # Ideal total head of the pump, m
                                 "H_static_ideal": None,  # Ideal static head of the pump, m
                                 "flow_coefficient_inlet": None, # Flow coefficient at the inlet of the impeller, -
                                 "flow_coefficient_outlet": None, # Flow coefficient in the diffuser throat, -
                                 "static_head_coefficient": None, # Static head coefficient of the pump, -
                                 "P_h_useful": None, # Useful pump power, W
                                 "P_h_losses": None, # Hydraulic power losses, W
                                 "P_h_total": None,  # Total hydraulic power, W
                                 "P_total": None,  # Total pump power, W
                                 "P_f": None,  # Power to overcome friction, W
                                 "eta_total": None, # Pump total efficiency, -
                                 "eta_static": None,  # Pump total efficiency, -
                                 "eta_losses": None, # Fraction of dynamic head lost in the diffuser, -
                                 "T_upstream": None, # Upstream temperature, K
                                 "rho": None, # Fluid density used for the analysis, kg/m^3
                                 "p_upstream": None,  # Upstream total (tank/reservoir) pressure, Pa
                                 "p_inlet": None, # Pressure at the impeller eye, Pa
                                 "p_2": None, # Pressure at the impeller outlet, Pa
                                 "p_4": None, # Pressure at the diffuser outlet, Pa
                                 "v_inlet": None,  # Velocity in the inlet pipe, m/s
                                 "v_0": None, # Velocity at the impeller eye, m/s
                                 "u_1": None, # Blade velocity at the impeller inlet, m/s
                                 "v_1ax": None,  # Axial velocity at the impeller inlet, m/s
                                 "v_1m": None,  # Meridional velocity at the impeller inlet, m/s
                                 "v_1": None,  # Absolute velocity at the impeller inlet, m/s
                                 "u_2": None, # Blade velocity at the impeller outlet, m/s
                                 "v_3": None, # Fluid velocity in the diffuser throat, m/s
                                 "v_4": None, # Fluid velocity at the diffuser outlet, m/s
                                 "T_4": None,  # Outlet temperature, K
                                 }

        # Another dictionary to store analysis results at the design point
        self.analysis_results_design = copy.deepcopy(self.analysis_results)

        # Constants
        self.g = 9.80665 # gravitational acceleration m/s^2
        self.CMPS_to_GPM = 15850.323140625 # conversion factor between cubic meters per second and gallons per minute
        self.feet_to_m = 0.3048 # conversion factor from feet to meter, ft/m
        self.lb_to_kg = 0.45359237 # conversion factor from pound to kilograms, kg/lb
        self.inch_to_m = 0.0254 # conversion factor from inch to meter, inch/m
        self.HP_to_W = 745.699872  # conversion factor from horsepower to Watt, W/HP

        # Digitized graphs from "A Forced Vortex Pump for High Speed, High Pressure, Low Flow Applications" by Lock.
        # Used for head analysis with his method.
        # First interpolator for h_0 graph for radial blades (upper left most graph from Figure 10a)
        r_ratio = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
        n_blades = np.array([2, 4, 8, 16])
        h_0_2_blades = np.array([0.498, 0.482, 0.435, 0.334, 0.190, 0])
        h_0_4_blades = np.array([0.636, 0.636, 0.606, 0.517, 0.332, 0])
        h_0_8_blades = np.array([0.768, 0.768, 0.768, 0.731, 0.518, 0])
        h_0_16_blades = np.array([0.864, 0.864, 0.864, 0.864, 0.753, 0])
        h_0 = np.transpose([h_0_2_blades, h_0_4_blades, h_0_8_blades, h_0_16_blades])
        # 2D interpolator for number of blades <= 16
        h_0_interp_2D = intrp.RegularGridInterpolator(points=(r_ratio, n_blades), values=h_0, method="pchip")
        # 1D interpolator for number of blades > 16
        h_0_interp_16 = intrp.PchipInterpolator(r_ratio, h_0_16_blades)
        # Property to store interpolating function
        self.h_0 = lambda r, n: (h_0_interp_2D([[r, n]])[0] if n <= 16 else h_0_interp_16(r))
        # Now interpolator for C_h graph for radial blades (upper line from Figure 10b)
        blade_spacing_length_ratio = [0.506, 0.606, 0.704, 0.804, 0.904, 1.003, 1.104, 1.202, 1.301, 1.401, 1.501,
                                      1.603, 1.704, 1.803, 1.903, 2.003, 2.104, 2.206, 2.307, 2.406, 2.505]
        C_h = [0.995, 0.994, 0.988, 0.980, 0.967, 0.955, 0.942, 0.926, 0.908, 0.890, 0.872, 0.854, 0.837, 0.819, 0.803,
               0.787, 0.772, 0.757, 0.743, 0.729, 0.717]
        C_h_interp = intrp.interp1d(blade_spacing_length_ratio, C_h, kind="linear", bounds_error=False, fill_value="extrapolate")
        # Property to store interpolating function
        self.C_h = lambda sigma: np.clip(C_h_interp(sigma), 0.0, 1.0)

    def size_dimensions(self, fluid, RPM, dp, mdot, p_upstream, T_upstream, inlet_sizing_method, diameter_sizing_method,
                        widths_sizing_method, outlet_sizing_method, t_hub, t_LE, t_TE, D_inlet, n_blades=5,
                        diffuser_area_ratio = 4, diffuser_angle = 8, flow_coefficient_outlet=0.8, D_1_over_D_0=1.1,
                        D_5_over_D_1 = 1.1, alpha_1 = 90, v_0=3.6576, u_1=45.72, flow_coefficient_inlet=0.07,
                        L_1_over_D_1=0.25, r_factor = 0.8, eta_losses=0.194, K_factor=0.17, no_prerotation=False,
                        D_diffuser_outlet=None):
        """A method to size the Barske pump. It updates parameters of the BarskePump object.

        :param fluid: Object representing fluid used for the sizing of the Barske Pump.
        :param float or int RPM: Design Rotations Per Minute at BEP, 1 / minute
        :param float or int dp: Pressure rise across the pump, Pa
        :param float or int mdot: Massflow through the pump, kg/s
        :param float or int p_upstream: Upstream total (tank/reservoir) pressure, Pa
        :param float or int T_upstream: Upstream temperature, K
        :param string inlet_sizing_method: Method used for sizing of the pump inlet, either "Lobanoff", "flow velocity",
            "blade velocity" or "flow coefficient". If "Lobanoff", eq.11-22 from 'Centrifugal Pumps' by Lobanoff et al.
            will be used for sizing impeller eye diameter D_0. If "flow velocity", D_0 will be sized based on assigned
            velocity v_0. In case of both of these options, impeller inlet diameter D_1 will be calculated from
            D_0 and D_1_over_D_0. If "blade velocity", D_1 is calculated from assigned blade velocity, u_1.
            If "flow coefficient", it is calculated from assigned flow_coefficient_inlet.
            D_0 will be then size from D_1 and D_1_over_D_0.
        :param string diameter_sizing_method: Method used for sizing of the impeller outer diameter D_2.
            It should be either "Lobanoff" or "Lock". If "Lobanoff", eq. 11-4 from 'Centrifugal Pumps'
            by Lobanoff et al. will be used. If "Lock", procedure from "A Forced Vortex Pump for High Speed,
            High Pressure, Low Flow Applications" by Lock will be used.
        :param string widths_sizing_method: Method used for sizing of the pump widths (axial lengths), either
            "Gulich", "Rocketdyne" or "diameter fraction". If "Gulich", impeller outlet width L_2 is calculated from
             equation 7-1a from "Centrifugal Pumps" by Gulich. L_1 is then calculated such that constant meridional
             velocity is achieved. If "Rocketdyne", then impeller inlet width L_1 is calculated from eq. 26 from
            "Rotating And Positive-Displacement For Low-Thrust Rocket Engines Pumps" by Rocketdyne.
             If "diameter fraction", L_1 is calculated from assigned L_1_over_D_1. In both of these options, L_2 is then
             calculated from L_1 such that constant meridional velocity is achieved.
        :param string outlet_sizing_method: Method used for sizing the diffuser, either "area ratio" or "outlet diameter".
             If "area ratio", diffuser_area_ratio will be used to calculate outlet diameter. If "outlet diameter", it
             will be calculated directly.
        :param float or int t_hub: Hub thickness, m
        :param float or int t_LE: Leading edge (suction side) thickness, m
        :param float or int D_inlet: Inlet pipe diameter, m
        :param float or int t_TE: Trailing edge (suction side) thickness, m
        :param int n_blades: Number of blades. By default, 5, which is within a range of 3-6 often mentioned for Barske
            impellers.
        :param float or int diffuser_area_ratio: Area ratio of the conical diffuser. Must be greater than 1.
        :param float or int diffuser_angle: Full angle of the conical diffuser in degrees. By default, 8 degrees, which
            is a common value for diffusers.
        :param float or int flow_coefficient_outlet: Flow coefficient of the pump at its outlet, that is, the ratio of
            fluid velocity in the throat v_3 to the outlet blade speed u_2. By default, 0.8.
        :param float or int D_1_over_D_0: Ratio of D_1 to D_0. By default, 1.1.
        :param float or int D_5_over_D_1: Ratio of D_5 to D_1. By default, 1.1.
        :param float or int alpha_1: impeller blade backward edge angle wrt. rotation axis, degrees. By default,
            90 degrees.
        :param float or int v_0: Impeller eye axial velocity, used in "flow velocity" option for inlet_sizing_method.
            By default, 3.6576 m/s. Default value taken from value of 12 ft/s, which is the upper recommended limit by
             Barske in "The Design of Open Impeller Centrifugal Pumps".
        :param float or int u_1: Impeller inlet blade speed, used in "blade velocity" option for inlet_sizing_method.
            By default, 45.72 m/s. Default value taken from value of 150 ft/s, which is the upper recommended limit by
             Barske in "The Design of Open Impeller Centrifugal Pumps".
        :param float or int flow_coefficient_inlet: Flow coefficient at the inlet, used in "flow coefficient" option for
            inlet_sizing_method. It is the ratio of the axial fluid velocity at the impeller inlet v_1ax to the inlet
            blade speed u_1. By default, 0.07, which is the recommended value in "Rotating And Positive-Displacement For
            Low-Thrust Rocket Engines Pumps" by Rocketdyne.
        :param float or int L_1_over_D_1: Ratio of L_1 to D_1, used in "diameter fraction" option for widths_sizing_method.
            By default, 0.25. This is the lower recommended limit  by Barske in "The Design of Open Impeller Centrifugal
            Pumps".
        :param float or int r_factor: Semi-empirical factor used in "Rocketdyne" option for widths_sizing_method, which
            is the flow area to meridional area ratio. By default, a value of 0.8 recommended by Rocketdyne.
        :param float or int eta_losses: Fraction of dynamic head lost in the diffuser, -. By default, 0.194. This is the
            value used by Lock in "A Forced Vortex Pump for High Speed, High Pressure, Low Flow Applications"
        :param float or int K_factor: Factor for prerotation at zero flow as a fraction of inlet tip speed, used for
            "Lock" option for diameter_sizing_method. By default, 0.17. Middle value from the range used in
             "A Forced Vortex Pump for High Speed, High Pressure, Low Flow Applications".
         :param bool no_prerotation: Determines where no prerotation should be assumed for impeller head calculations
            when diameter_sizing_method is "Lobanoff", True or False. By default, False, as recommended in Lobanoff.
        """
        # Get fluid density
        rho = fluid.get_density(p_upstream, T_upstream) # kg/s

        # Calculate volumetric flow and the required head. When calculating volumetric flow, it is assumed fluid is
        # incompressible.
        Q_design = mdot / rho  # m^3 / s
        H_required = functions.get_H_from_dp(fluid, dp, p_upstream, T_upstream)

        # Calculate pressure in the inlet pipe
        self.D_inlet = D_inlet
        p_static_inlet = p_upstream - 0.5 * rho * (4 * Q_design/(np.pi * self.D_inlet**2))**2

        # Calculate specific speed and assign number of blades
        self.specific_speed = RPM * np.sqrt(Q_design) / (H_required**0.75)
        self.n_blades = n_blades

        # Calculate angular speed of rotation
        omega = RPM * 2 * np.pi / 60 # rad / s

        # Size the inlet diameters
        self.D_1_over_D_0 = D_1_over_D_0
        # If inlet sizing method is 'Lobanoff' or 'flow velocity', first size the eye diameter and then impeller inlet
        # diameter
        if inlet_sizing_method in ("Lobanoff", "flow velocity"):
            if inlet_sizing_method == "Lobanoff":
                self.D_0 = 5.1 * (Q_design * self.CMPS_to_GPM / RPM)**0.333 # m
            elif inlet_sizing_method == "flow velocity":
                self.D_0 = np.sqrt(4 * Q_design / (np.pi * v_0))    # m
            self.D_1 = self.D_0 * self.D_1_over_D_0 # m
        # If inlet sizing method is inlet flow coefficient or inlet blade speed, first size the impeller inlet diameter
        # and then the eye diameter
        elif inlet_sizing_method in ("blade velocity", "flow coefficient"):
            if inlet_sizing_method == "blade velocity":
                self.D_1 = 2 * u_1 / omega  # m
            elif inlet_sizing_method == "flow coefficient":
                self.D_1 = (8 * Q_design / (np.pi * omega * flow_coefficient_inlet))**(1/3) # m
            self.D_0 = self.D_1 / self.D_1_over_D_0 # m
        # If wrong option was assigned, raise error
        else:
            warnings.simplefilter("error", UserWarning)
            warnings.warn("inlet_sizing_method must be 'Lobanoff', 'flow velocity', 'blade velocity' or"
                          " 'flow coefficient'")

        # Size hub diameter. If alpha_1 is not 90 degrees and D_5_over_D_1 is different from 1, it will be changed to 1.
        if alpha_1 != 90 and D_5_over_D_1 != 1:
            warnings.warn("Alpha_1 is not 90 degrees. To ensure feasible geometry, D_5_over_D_1 is set to 1.")
            D_5_over_D_1 = 1
        self.D_5_over_D_1 = D_5_over_D_1
        self.D_5 = self.D_5_over_D_1 * self.D_1 # m

        # Assign thicknesses and other known quantities
        self.t_0 = t_hub # m
        self.t_1 = t_LE # m
        self.t_2 = t_TE # m
        self.alpha_1 = alpha_1 # degrees
        self.alpha_diffuser = diffuser_angle # degrees
        self.eta_losses_design = eta_losses # -
        self.K_factor_design = K_factor # -
        self.flow_coefficient_BEP = flow_coefficient_outlet # -
        self.no_prerotation_design = no_prerotation # -

        #TODO Take thicknesses into account
        #TODO Incorporate splitter blades

        # Find outlet diameter that satisfies requirements
        # First define a function that calculates impeller's dimensions and H as a function of D_2
        def get_impeller_head(D_2):
            # Calculate blade tip speed
            u_2 = D_2 * omega / 2 # m/s

            # Size the diffuser from the assigned outlet flow coefficient
            v_3 = u_2 * self.flow_coefficient_BEP # m/s
            D_3 = np.sqrt(4 * Q_design / (np.pi * v_3)) # m
            A_3 = np.pi * (D_3 / 2)**2 # m^2
            # If diffuser sizing is done through area ratio
            if outlet_sizing_method == "area ratio":
                A_4 = self.A_4_over_A_3 * A_3  # m^2
                D_4 = 2 * np.sqrt(A_4 / np.pi)  # m
            # If it is done with outlet diameter
            elif outlet_sizing_method == "outlet diameter":
                if D_diffuser_outlet is None:
                    warnings.simplefilter("error", UserWarning)
                    warnings.warn("D_diffuser_outlet cannot be None if outlet_sizing_method is 'outlet diameter'.")
                D_4 = D_diffuser_outlet # m
                A_4 = np.pi * (D_4 / 2)**2 # m2
            # If wrong input was given
            else:
                warnings.simplefilter("error", UserWarning)
                warnings.warn("outlet_sizing_method must be 'area ratio' or 'outlet diameter'.")
            diffuser_area_ratio = A_4 / A_3
            L_diffuser = ((D_4 - D_3) / 2) / np.tan(self.alpha_diffuser * np.pi / 180) # m

            # If blade width sizing method is Gulich, first get L_2 and then get L_1
            if widths_sizing_method == "Gulich":
                L_2 = (0.02 + 0.5 * self.specific_speed/100 - 0.03 * (self.specific_speed/100)**2
                       - 0.04 * (self.specific_speed/100)**3) * D_2 # m
                L_1 = D_2 * L_2 / self.D_1 # m
            # If it is Barske or Rocketdyne method, first get L_1 and then get L_2
            elif widths_sizing_method in ("Barske", "Rocketdyne"):
                if widths_sizing_method == "Barske":
                    L_1 = L_1_over_D_1 * self.D_1   # m
                elif widths_sizing_method == "Rocketdyne":
                    L_1 = np.pi * self.D_1 / (4 * r_factor) # m
                L_2 = self.D_1 * L_1 / D_2  # m
            else:
                warnings.simplefilter("error", UserWarning)
                warnings.warn("widths_sizing_method must be 'Gulich', 'Barske' or 'Rocketdyne'")

            # Calculate the rest of the geometry. First calculate alpha_0, which is the angle of the forward edge of
            # the impeller wrt. axis of rotation. dy_TE is how much trailing edge is lifted wrt. the hub.
            dy_TE = np.tan((90 - self.alpha_1) * np.pi / 180) * ((D_2 - self.D_1) / 2) # m
            alpha_0 = np.arctan(((D_2 - self.D_1) / 2) / (L_1 + self.t_0 - L_2 - dy_TE)) * 180 / np.pi # degrees
            # Now calculate the sharpening angle of the suction side wrt. tangent of D_1. It is the same as local flow
            # angle. It is assumed that the fluid flows through whole perimeter of D_1.
            alpha_2 = np.arctan((Q_design / (L_1 * np.pi * self.D_1)) / u_1) * 180 / np.pi

            # Calculate axial and radial clearances between the casing and the impeller. From Barske (pg. 7 of
            # "The Design of Open Impeller Centrifugal Pumps"), it should be 1% of D_2, but should not be greater than
            # 0.04 inch of larger pumps. From the empirical data from the same page, a linear equation for radial
            # clearance is derived.
            s_ax = min(0.01 * D_2, 0.04 * 0.0254)
            s_rad = self.calcualate_axial_clearance(D_2)

            # Analyse performance either with Lobanoff or Lock method
            if diameter_sizing_method == "Lock":
                H = self.analysis_Lock(u_2, Q_design, fluid, p_static_inlet, T_upstream, self.eta_losses_design,
                                       self.K_factor_design)[0]
            elif diameter_sizing_method == "Lobanoff":
                H = self.analysis_Lobanoff(u_2, u_1, Q_design)
            else:
                warnings.simplefilter("error", UserWarning)
                warnings.warn("diameter_sizing_method must be 'Lock' or 'Lobanoff'")

            return H, u_2, D_3, D_4, A_3, A_4, diffuser_area_ratio, L_diffuser, L_1, L_2, alpha_0, alpha_2, s_ax, s_rad

        # Now solve the function for D_2 and get all other remaining dimensions
        D_2_estimate = np.sqrt((4 * self.g * H_required) / (0.7 * omega**2))
        self.D_2 = opt.toms748(f=lambda x: H_required - get_impeller_head(x), a=self.D_1 * 1.1, b=2*D_2_estimate)[0]
        (H_design, u_2, self.D_3, self.D_4, self.A_3, self.A_4, self.A_4_over_A_3, self.L_diffuser, self.L_1, self.L_2,
         self.alpha_0, self.alpha_2, self.s_ax, self.s_rad) = get_impeller_head(self.D_2)

        # Regardless with what method diameter was sized, obtain static head coefficient for Lobanoff method.
        # This is necessary in case the sized geometry is analyzed in the future with methods proposed by Lobanoff in
        # "Centrifugal Pumps" or Barske in "The Design of Open Impeller Centrifugal Pumps". The head coefficient
        # follows the convention in "Centrifugal Pumps" by Gulich.
        self.static_head_coefficient_BEP_Barske =\
            self.analysis_Lobanoff(u_2, u_1, Q_design)[2]

        # For informational purposes, obtain static head coefficient for Lock's method as well
        self.static_head_coefficient_BEP_Lock =\
            self.analysis_Lock(u_2, Q_design, fluid, p_static_inlet, T_upstream, self.eta_losses_design,
                               self.K_factor_design)[5]

        # With all dimensions known, full analysis can be performed to get more data about flow and performance
        if diameter_sizing_method is "Lobanoff":
            analysis_method = "Lobanoff"
        else:
            analysis_method = "Barske"
        self.analysis_results_design = self.analyse(fluid, mdot, RPM, p_upstream, T_upstream, analysis_method,
                                                    K_factor, eta_losses, no_prerotation)

        # Finally, design can be verified
        self.verify_design()

    def analysis_Lock(self, u_2, Q, fluid, p_0, T_0, eta_losses, K_factor):
        #TODO add docstring.

        # First find h_0 and C_h factors from digitalized Figure 10a and Figure 10b from Lock. To do so, ratio of
        # impeller radii and blade spacing to length ratio need to be found.
        r_ratio = self.D_2 / self.D_1
        blade_spacing_length_ratio = (self.D_1 * np.pi / self.n_blades) / ((self.D_2 - self.D_1) / 2)
        h_0 = self.h_0(r_ratio, self.n_blades)
        C_h = self.C_h(blade_spacing_length_ratio)

        # Now determine Q_ops for which peak power value occurs.
        dummy_1a = (h_0 * u_2**2) / self.g
        dummy_1b = eta_losses * 24 * (self.g * self.D_3**4 * np.pi**2)**(-1) * (1 - (self.D_3 / self.D_4)**2)**2
        dummy_1c = 24 * (self.D_4**(-4) - self.D_inlet**(-4)) / (np.pi**2 * self.g)
        Q_ops = np.sqrt(dummy_1a / (dummy_1b + dummy_1c)) # m^3 / s

        # Now calculate ideal total head
        H_total_ideal = dummy_1a - C_h * K_factor * ((u_2**2) / self.g) * (self.D_1 / self.D_2) * (1 - (Q / Q_ops))
        # Now calculate ideal static head
        H_static_ideal = H_total_ideal + (- dummy_1c / 3) * Q**2
        # Calculate diffuser losses
        H_loss = (dummy_1b / 3) * Q**2
        # Then calculate real total head
        H_total_real = H_total_ideal - H_loss
        # Now calculate real static head
        H_static_real = H_static_ideal - H_loss
        # Also calculate static head due to forced vortex. This is total head minus impellet exit velocity
        H_s = H_total_ideal - ((u_2**2) / (2 * self.g))

        # Also determine whether Q is above the maximum volumetric flow. First determine head at vapor pressure wrt.
        # inlet pressure.
        H_vp = functions.get_H_from_dp(fluid=fluid, pressure_rise=fluid.get_vp(T_0) - p_0, p_0=p_0, T_0=T_0)
        # Now determine the head in the throat wrt. inlet pressure
        H_3 = H_static_real + H_loss - (8 * Q**2) * (self.D_3**(-4) - self.D_4**(-4)) / (self.g * np.pi**2)
        # If that head is equal or smaller than H_3, the H-Q curve breaks down according to Lock.
        if H_3 <= H_vp:
            # H_loss becomes the total ideal head
            H_loss = H_total_ideal
            # Total real head becomes zero
            H_total_real = 0
            # Static head is then just the difference in dynamic heads
            H_static_real = (- dummy_1c / 3) * (Q**2)

        # Calculate head coefficients (as defined in Gulich)
        head_coefficient_static = 2 * self.g * H_static_real / (u_2 ** 2)
        head_coefficient_total = 2 * self.g * H_total_real / (u_2 ** 2)

        # Now return all the results
        return (H_static_real, H_total_real, H_loss, H_total_ideal, H_static_ideal, head_coefficient_static,
                head_coefficient_total, H_s)

    def analysis_Lobanoff(self, u_2, u_1, Q):
        # TODO add docstring.
        # Calculate head due to forced vortex
        if self.no_prerotation_design:
            Hs = (u_2**2 - u_1**2) / (2 * self.g)
        else:
            Hs = u_2**2 / (2 * self.g)
        # Now calculate the static recovery from the dynamic head
        Hd = (1 - self.eta_losses_design) * ((u_2 * self.flow_coefficient_BEP)**2 / (2 * self.g)) * \
             (1 - (self.A_3 / self.A_4)**2)
        # Now calculate the real total head
        v_4 = 4 * Q / (np.pi * self.D_4**2)
        H_total_real = Hs + Hd + (v_4**2) / (2 * self.g)
        # Finally calculate the real static head
        v_inlet = 4 * Q / (np.pi * self.D_inlet**2)
        H_static_real = H_total_real - (v_4**2) / (2 * self.g) + (v_inlet**2) / (2 * self.g)
        # Calculate the total and static head coefficients (as defined in Gulich)
        head_coefficient_static = 2 * self.g * H_static_real / (u_2**2)
        head_coefficient_total = 2 * self.g * H_total_real / (u_2**2)
        return H_static_real, H_total_real, head_coefficient_static, head_coefficient_total
    
    def analysis_Barske(self, u_1, u_2, v_inlet, v_3, v_4, no_prerotation):
        # TODO add docstring.

        # Calculate ideal head due to forced vortex
        if no_prerotation:
            H_s = (u_2**2 - u_1**2) / (2 * self.g)
        else:
            H_s = u_2**2 / (2 * self.g)
        # Calculate ideal dynamic head
        H_d = u_2**2 / (2 * self.g)
        # Calculate ideal total and static heads
        H_total_ideal = H_s + H_d
        H_static_ideal = H_total_ideal + ((v_inlet**2 - v_4**2) / (2 * self.g))

        # Calculate real static head. If the same prerotation assumption is used for analysis as in the case
        # of impeller design, use already calculated static head coefficient
        if no_prerotation == self.no_prerotation_design:
            static_head_coefficient = self.static_head_coefficient_BEP_Barske
        # If no prerotation was assumed for the design, but analysis assumes prerotation, then static head coefficient
        # needs to be increased
        elif self.no_prerotation_design and no_prerotation is not self.no_prerotation_design:
            static_head_coefficient = self.static_head_coefficient_BEP_Barske + ((u_1/u_2)**2)
        # If prerotation was assumed for the design, but analysis assume no prerotation, then static head coefficient
        # needs to be decreased
        elif self.no_prerotation_design is False and no_prerotation:
            static_head_coefficient = self.static_head_coefficient_BEP_Barske - ((u_1/u_2)**2)

        # Calculate real static head
        H_static_real = static_head_coefficient
        # Calculate real total head
        H_total_real = H_static_real + ((v_4**2 - v_inlet**2) / (2 * self.g))
        # Calculate head loss
        H_loss = H_total_ideal - H_total_real

        # Calculate maximum velocity. Barske assumes maximum flow is reached when all pump head is equal to the dynamic
        # head in the throat
        v_max = np.sqrt(2 * self.g * H_total_real)
        # If velocity in the throat is above the maximum value, the H-Q curve breaks down
        if v_max > v_3:
            # H_loss becomes the total ideal head
            H_loss = H_total_ideal
            # Total real head becomes zero
            H_total_real = 0
            # Static head is then just the difference in dynamic heads
            H_static_real = ((v_inlet**2 - v_4**2) / (2 * self.g))

        # Recalculate flow coefficients
        head_coefficient_static = 2 * self.g * H_static_real / (u_2 ** 2)
        head_coefficient_total = 2 * self.g * H_total_real / (u_2 ** 2)

        return (H_static_real, H_total_real, H_loss, H_total_ideal, H_static_ideal, head_coefficient_static,
                head_coefficient_total, H_s)

    def calcualate_axial_clearance(self, D_2):
        # TODO add docstring.
        return (0.008 * (D_2 / self.inch_to_m) + 0.072) * self.inch_to_m

    def analyse(self, fluid, mdot, RPM, p_upstream, T_upstream, analysis_method, K_factor=None, eta_losses=None,
                no_prerotation=None):
        """A method to analyze pump's performance for given geometry"""
        #TODO: Update docstring

        # If eta_losses is None, use the same value as the one used for design of the impeller
        if eta_losses is None: eta_losses = self.eta_losses_design
        # If K_factor is None, use the same value as the one used for design of the impeller
        if K_factor is None: K_factor = self.K_factor_design
        # If no_prerotation is None, use the same boolean as for the design
        if no_prerotation is None: no_prerotation = self.no_prerotation_design

        # Get fluid properties
        rho = fluid.get_density(p_upstream, T_upstream)  # kg/s
        kinematic_viscosity = fluid.get_kinematic_viscosity(p_upstream, T_upstream) # m^2 / s
        # Calculate volumetric flow at the inlet
        Q = mdot / rho  # m3 / s
        # First calculate flow speed and static pressure at the inlet.
        v_inlet = 4 * Q / (np.pi * self.D_inlet**2)
        p_inlet = p_upstream - 0.5 * rho * v_inlet**2
        # Calculate flow velocity at the impeller eye
        v_0 = 4 * Q / (np.pi * self.D_0**2)
        # Then calculate u_1, v_1m and v_1ax
        omega = RPM * 2 * np.pi / 60 # rad / s
        u_1 = omega * self.D_1 / 2
        v_1ax = 4 * Q / (np.pi * self.D_1**2)
        v_1m = Q / (np.pi * self.D_1 * self.L_1)
        v_1 = np.sqrt(u_1**2 + v_1m**2)
        # Calculate u_2
        u_2 = omega * self.D_2 / 2
        # Calculate v_3 and v_4
        v_3 = 4 * Q / (np.pi * self.D_3 ** 2)
        v_4 = 4 * Q / (np.pi * self.D_4 ** 2)

        # Calculate paddle power to overcome friction
        dummy_1a = 0.6e-6 * (rho / self.lb_to_kg) * (self.feet_to_m**3)
        dummy_1b = ((kinematic_viscosity  / (self.feet_to_m**2))**0.2) * ((RPM / 1000)**2.8)
        dummy_1c = (np.sin(self.alpha_0 * np.pi / 180)**(-1)) + (np.sin(self.alpha_1 * np.pi / 180)**(-1))
        dummy_1d = (self.D_2 / self.inch_to_m)**4.6
        dummy_1e = 9.2 * ((self.D_1 / self.inch_to_m)**3.6) * (self.L_1 / self.inch_to_m)
        P_f = (dummy_1a * dummy_1b * (dummy_1c * dummy_1d + dummy_1e)) * self.HP_to_W # W
        
        # Calculate flow coefficient
        flow_coefficient_inlet = v_1ax / u_1
        flow_coefficient_outlet = v_3 / u_2

        # Calculate impeller ideal heads and coefficients depending on the method chosen
        if analysis_method is "Lock":
            (H_static_real, H_total_real, H_loss, H_total_ideal, H_static_ideal, head_coefficient_static,
             head_coefficient_total, H_s) = self.analysis_Lock(u_2, Q, fluid, p_inlet, T_upstream, eta_losses, K_factor)
        elif analysis_method is "Barske":
            (H_static_real, H_total_real, H_loss, H_total_ideal, H_static_ideal, head_coefficient_static,
             head_coefficient_total, H_s) = self.analysis_Barske(u_1, u_2, v_inlet, v_3, v_4, no_prerotation)
        else:
            warnings.simplefilter("error", UserWarning)
            warnings.warn("analysis_method must be 'Lock' or 'Barske'")

        # Calculate p_2. To get difference between static head at the inlet and static head at the outlet, dynamic head
        # at the inlet must be added to static head due to forced vortex in the impeller
        p_2 = p_inlet + functions.get_dP_from_H(fluid, H_s + ((v_inlet**2) / (2 * self.g)), p_inlet, T_upstream)
        # Calculate p_4.
        dp = functions.get_dP_from_H(fluid, H_static_real, p_inlet, T_upstream)
        p_4 = p_inlet + dp

        # Calculate useful pump power
        P_h_useful = mdot * H_total_real * self.g
        # Calculate pump power lost to hydraulic losses
        P_h_losses = mdot * H_loss * self.g
        # Calculate total hydraulic pump power
        P_h_total = mdot * H_total_ideal * self.g
        # Calculate total pump power
        P_total = P_h_total + P_f
        # Calculate pump efficiency
        eta_static = mdot * H_static_real * self.g / P_total
        eta_total = P_h_total / P_total
        # Calculate outlet temperature
        T_outlet = T_upstream + ((P_h_losses + P_f) / (mdot * fluid.get_Cp(p_inlet, T_upstream)))

        # Pack results into dictionary and return it
        analysis_results = {"method": analysis_method, "fluid": fluid, "RPM": RPM, "omega": omega, "mdot": mdot,
                            "dp": dp, "Q": Q, "H_total_real": H_total_real, "H_static_real": H_static_real,
                            "H_losses": H_loss, "H_total_ideal": H_total_ideal, "H_static_ideal": H_static_ideal,
                            "flow_coefficient_inlet": flow_coefficient_inlet,
                            "flow_coefficient_outlet": flow_coefficient_outlet,
                            "static_head_coefficient": head_coefficient_static, "P_h_useful": P_h_useful,
                            "P_h_losses": P_h_losses, "P_h_total": P_total, "P_total": P_total, "P_f": P_f,
                            "eta_total": eta_total, "eta_static": eta_static, "eta_losses": eta_losses,
                            "T_upstream": T_upstream, "p_upstream": p_upstream, "rho": rho, "p_inlet": p_inlet,
                            "p_2": p_2, "p_4": p_4, "v_inlet": v_inlet, "v_0": v_0, "u_1": u_1, "v_1ax": v_1ax,
                            "v_1m": v_1m, "v_1": v_1, "u_2": u_2, "v_3": v_3, "v_4": v_4, "T_4": T_outlet}
        return analysis_results

    def verify_design(self):
        """A method to verify the design of the Barske pump"""
        # Verify if axial velocity at the inlet is within range given by Barske
        if not (5 * self.feet_to_m <= self.analysis_results_design["v_0"] <= 12 * self.feet_to_m):
            print(f"Impeller eye diameter v0 is {self.analysis_results_design["v_0"]} m/s."
                  f" It should be between {5*self.feet_to_m} m/s and {12*self.feet_to_m} m/s.")
        # Verify if inner blade velocity is smaller as recommended by Barske
        if self.analysis_results_design["u_1"] > 150 * self.feet_to_m:
            print(f"Inner blade speeed u1 is {self.analysis_results_design["u_1"]} m/s."
                  f" It should be <= {150 * self.feet_to_m} m/s.")
        # Verify if diameter ratio is greater as recommended by Barske
        if self.D_2 / self.D_1 <= 1.5:
            print(f"Diameter ratio D2/D1 is {self.D_2 / self.D_1}."
                  f"It should be > 1.5.")
        # Verify axial widths
        if self.L_1 < 0.25 * self.D_1:
            print(f"Axial width L1 at impeller inlet is {self.L_1 * 1000} mm."
                  f" It should be above {0.25 * self.D_1 * 1000} mm.")
        if self.L_2 < self.L_1 * self.D_1 / self.D_2:
            print(f"Axial width L2 at impeller outlet is {self.L_2 * 1000} mm."
                  f" It should be above {self.L_1 * self.D_1 / self.D_2 * 1000} mm.")
        # Verify that clearances are within values recommended by Barske
        if self.s_rad > min(0.01 * self.D_2, 0.04 * 0.0254):
            print(f"Radial clearance s_rad is {self.s_rad * 1000} mm."
                  f" It should be below {min(0.01 * self.D_2, 0.04 * 0.0254) * 1000} mm.")
        if self.s_ax > self.calcualate_axial_clearance(self.D_2):
            print(f"Axial clearance s_ax is {self.s_ax * 1000} mm."
                  f" It should be below {self.calcualate_axial_clearance(self.D_2) * 1000} mm")
        if self.L_2 < 3 * self.s_ax:
            print(f"Axial width L2 at impeller outlet is {self.L_2 * 1000} mm."
                  f" It should be greater than 3s_ax = {3 * self.s_ax * 1000} mm.")

    def print_dimensions(self):
        """A method to print pump's dimensions in a GitHub-style table"""

        rows = [
            # Diameters
            ["D_inlet", self.D_inlet * 1e3, "mm", "Inlet pipe diameter"],
            ["D_0", self.D_0 * 1e3, "mm", "Impeller eye diameter"],
            ["D_1", self.D_1 * 1e3, "mm", "Impeller inlet diameter"],
            ["D_2", self.D_2 * 1e3, "mm", "Impeller outlet diameter"],
            ["D_3", self.D_3 * 1e3, "mm", "Diffuser throat diameter"],
            ["D_4", self.D_4 * 1e3, "mm", "Diffuser outlet diameter"],
            ["D_5", self.D_5 * 1e3, "mm", "Hub diameter"],
            ["D_1/D_0", self.D_1_over_D_0, "-", "Ratio of D1 to D0"],
            ["D_5/D_1", self.D_5_over_D_1, "-", "Ratio of D5 to D1"],

            # Widths & lengths
            ["L_1", self.L_1 * 1e3, "mm", "Impeller inlet width"],
            ["L_2", self.L_2 * 1e3, "mm", "Impeller outlet width"],
            ["L_1/D_1", self.L_1_over_D_1, "-", "Ratio of L1 to D1"],
            ["L_diffuser", self.L_diffuser * 1e3, "mm", "Diffuser length"],

            # Areas
            ["A_0", self.A_0 * 1e6, "mm^2", "Impeller eye area"],
            ["A_3", self.A_3 * 1e6, "mm^2", "Diffuser throat area"],
            ["A_4", self.A_4 * 1e6, "mm^2", "Diffuser exit area"],
            ["A_4/A_3", self.A_4_over_A_3, "-", "Diffuser area ratio"],

            # Clearances
            ["s_ax", self.s_ax * 1e3, "mm", "Axial clearance"],
            ["s_rad", self.s_rad * 1e3, "mm", "Radial clearance"],

            # Thicknesses
            ["t_0", self.t_0 * 1e3, "mm", "Impeller hub thickness"],
            ["t_1", self.t_1 * 1e3, "mm", "Blade thickness at inlet"],
            ["t_2", self.t_2 * 1e3, "mm", "Blade thickness at outlet"],

            # Angles
            ["alpha_0", self.alpha_0, "deg", "Blade forward edge angle"],
            ["alpha_1", self.alpha_1, "deg", "Blade backward edge angle"],
            ["alpha_2", self.alpha_2, "deg", "Radial blade sharpening angle"],
            ["alpha_diffuser", self.alpha_diffuser, "deg", "Diffuser full angle"],

            # Other
            ["n_blades", self.n_blades, "-", "Number of blades"],
            ["specific_speed", self.specific_speed, "m,RPM,m^3/h", "Specific speed (EU)"],
        ]

        print(tabulate(
            rows,
            headers=["Parameter", "Value", "Unit", "Description"],
            tablefmt="github",
            floatfmt=".2f"
        ))

    def plot_geometry(self):
        """A method to plot geometry of the pump"""
        # Convert pump dimensions from meters to millimeters and from deg to rad
        D0 = self.D_0 * 1e3 #
        D1 = self.D_1 * 1e3
        D2 = self.D_2 * 1e3
        D3 = self.D_3 * 1e3
        D4 = self.D_4 * 1e3
        D5 = self.D_5 * 1e3
        L1 = self.L_1 * 1e3
        L2 = self.L_2 * 1e3
        L_diffuser = self.L_diffuser * 1e3
        t0 = self.t_0 * 1e3
        t1 = self.t_1 * 1e3
        t2 = self.t_2 * 1e3
        s_ax = self.s_ax * 1e3
        s_rad = self.s_rad * 1e3
        alpha2 = np.deg2rad(self.alpha_2)
        alpha1 = np.deg2rad(self.alpha_1)
        # Get radiuses
        r1 = D1 / 2
        r2 = D2 / 2
        r_eye = D0 / 2
        r_hub = D5 / 2
        r_volute = (D2 + 2 * s_rad) / 2

        # First create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Create base blade geometry at reference angular position, where x-axis is aligned with radial coordinates.
        # First get coordinates of the leading edge of the blade.
        blade_inner_point_1 = np.array([r1, -t1 / 2])
        # Coordinates of one of the LE points must be found from the sharpening angle. To do so, calculate slope of
        # blade suction surface
        blade_surface_slope = ((t2 / 2) - (t1 / 2)) / (r2 - r1)
        # Calculate slope of sharpened leading edge
        sharpening_slope = np.tan((np.pi/2) - alpha2)
        # Solve intersection
        blade_inner_point_2_x = r1 + (t1 / (sharpening_slope - blade_surface_slope))
        blade_inner_point_2_y = (-t1 / 2) + sharpening_slope * (blade_inner_point_2_x - r1)
        blade_inner_point_2 = np.array([
            blade_inner_point_2_x,
            blade_inner_point_2_y
        ])
        # Get coordinates of the outer points
        blade_outer_point_1 = np.array([r2, t2 / 2])
        blade_outer_point_2 = np.array([r2, -t2 / 2])
        #Assemble blade polygon
        blade_polygon = np.array([
            blade_inner_point_1,
            blade_inner_point_2,
            blade_outer_point_1,
            blade_outer_point_2,
            blade_inner_point_1
        ])
        # Create blade rotation function
        def rotate_points(points, angle):
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            return points @ rotation_matrix.T
        blade_pitch_angle = 2 * np.pi / self.n_blades
        rotation_offset = blade_pitch_angle / 2

        # Plot top view geometry
        ax_top = axes[0]
        theta = np.linspace(0, 2 * np.pi, 300)
        ax_top.plot(r_eye * np.cos(theta), r_eye * np.sin(theta), label="Impeller eye")
        ax_top.plot(r_hub * np.cos(theta), r_hub * np.sin(theta), label="Impeller hub")
        ax_top.plot(r_volute * np.cos(theta), r_volute * np.sin(theta), label="Pump volute")
        # Plot all blades
        for i in range(self.n_blades):
            angle = rotation_offset + i * blade_pitch_angle
            rotated_blade = rotate_points(blade_polygon, angle)
            if i == 0:
                ax_top.plot(rotated_blade[:, 0], rotated_blade[:, 1], label="Impeller blades")
            else:
                ax_top.plot(rotated_blade[:, 0], rotated_blade[:, 1])

        # Create diffuser geometry
        x0 = -r2
        y0 = 0
        throat_point_1 = np.array([x0 - (D3 / 2), y0])
        throat_point_2 = np.array([x0 + (D3 / 2), y0])
        outlet_point_1 = np.array([x0 - (D4 / 2), y0 + L_diffuser])
        outlet_point_2 = np.array([x0 + (D4 / 2), y0 + L_diffuser])
        # Create diffuser polygon
        diffuser_polygon = np.array([
            throat_point_2,
            outlet_point_2,
            outlet_point_1,
            throat_point_1
        ])
        # Plot the diffuser
        ax_top.plot(diffuser_polygon[:, 0], diffuser_polygon[:, 1], label="Pump diffuser")

        # Set graph options
        ax_top.set_title("Top view")
        ax_top.set_xlabel("x (mm)")
        ax_top.set_ylabel("y (mm)")
        ax_top.grid(True, which="major")
        ax_top.grid(True, which="minor", linestyle=":")
        ax_top.minorticks_on()
        ax_top.legend()
        ax_top.set_aspect("equal")

        # Plot side view geometry now in the next subfigure
        ax_side = axes[1]

        # Create blade geometry
        blade_side_point_0 = np.array([r1, 0])
        blade_side_point_1 = np.array([r1, t0 + L1])
        y2 = np.tan(alpha1) * (r2 - r1)
        blade_side_point_2 = np.array([r2, y2])
        blade_side_point_3 = np.array([r2, y2 + L2])
        # Create blade polygon
        blade_side_polygon = np.array([
            blade_side_point_0,
            blade_side_point_1,
            blade_side_point_3,
            blade_side_point_2,
            blade_side_point_0
        ])
        # Plot it
        ax_side.plot(blade_side_polygon[:, 0], blade_side_polygon[:, 1], label="Impeller blade")

        # Create pump casing geometry
        casing_point_0 = np.array([r1, -s_ax])
        casing_point_1 = np.array([r1, t0 + L1 + s_ax])
        casing_point_2 = np.array([r2 + s_rad, y2 - s_ax])
        casing_point_3 = np.array([r2 + s_rad, y2 + L2 + s_ax])
        # Create pump casing polygon
        casing_polygon = np.array([
            casing_point_0,
            casing_point_1,
            casing_point_3,
            casing_point_2
        ])
        # Plot it
        ax_side.plot(casing_polygon[:, 0], casing_polygon[:, 1], label="Pump casing")

        # Create impeller hub geometry
        hub_point_0 = np.array([0, 0])
        hub_point_1 = np.array([r_hub, 0])
        hub_point_2 = np.array([r_hub, t0])
        hub_point_3 = np.array([0, t0])
        # Create hub polygon
        hub_polygon = np.array([
            hub_point_0,
            hub_point_1,
            hub_point_2,
            hub_point_3
        ])
        # Plot it
        ax_side.plot(hub_polygon[:, 0], hub_polygon[:, 1], label="Impeller hub")

        # Set graph options
        ax_side.set_title("Side view")
        ax_side.set_xlabel("r (mm)")
        ax_side.set_ylabel("z (mm)")
        ax_side.grid(True, which="major")
        ax_side.grid(True, which="minor", linestyle=":")
        ax_side.minorticks_on()
        ax_side.legend()

        # Show figure
        plt.tight_layout()
        plt.show()

    def print_analysis_results(self):
        """A method to print results of the analysis"""

        r = self.analysis_results

        rows = [
            ["method", r["method"], "-", "Analysis method"],
            ["fluid", r["fluid"].get_name, "-", "Fluid used for analysis"],
            ["RPM", r["RPM"], "1/min", "Rotations per minute"],

            ["mdot", r["mdot"], "kg/s", "Mass flow rate"],
            ["Q", r["Q"], "m^3/s", "Volumetric flow rate"],

            ["dp", r["dp"] * 1e-5, "bar", "Pressure rise across pump"],

            ["flow_coefficient_inlet", r["flow_coefficient_inlet"], "-", "Flow coefficient at impeller inlet"],
            ["flow_coefficient_outlet", r["flow_coefficient_outlet"], "-", "Flow coefficient at diffuser throat"],
            ["static_head_coefficient", r["static_head_coefficient"], "-", "Static head coefficient"],

            ["H_total_real", r["H_total_real"], "m", "Real total head"],
            ["H_static_real", r["H_static_real"], "m", "Real static head"],
            ["H_losses", r["H_losses"], "m", "Head losses"],
            ["H_total_ideal", r["H_total_ideal"], "m", "Ideal total head"],
            ["H_static_ideal", r["H_static_ideal"], "m", "Ideal static head"],

            ["P_h_useful", r["P_h_useful"] * 1e-3, "kW", "Useful hydraulic power"],
            ["P_h_losses", r["P_h_losses"] * 1e-3, "kW", "Hydraulic losses power"],
            ["P_h_total", r["P_h_total"] * 1e-3, "kW", "Total hydraulic power"],
            ["P_total", r["P_total"] * 1e-3, "kW", "Total pump power"],
            ["P_f", r["P_f"] * 1e-3, "kW", "Friction power"],

            ["eta_total", r["eta_total"], "-", "Total efficiency"],
            ["eta_static", r["eta_static"], "-", "Static efficiency"],
            ["eta_losses", r["eta_losses"], "-", "Fraction of diffuser dynamic head lost"],

            ["T_upstream", r["T_upstream"], "K", "Upstream temperature"],
            ["rho", r["rho"], "kg/m^3", "Fluid density"],

            ["p_upstream", r["p_upstream"] * 1e-5, "bar", "Upstream pressure"],
            ["p_inlet", r["p_inlet"] * 1e-5, "bar", "Pressure at impeller eye"],
            ["p_2", r["p_2"] * 1e-5, "bar", "Pressure at impeller outlet"],
            ["p_4", r["p_4"] * 1e-5, "bar", "Pressure at diffuser outlet"],

            ["v_inlet", r["v_inlet"], "m/s", "Velocity in inlet pipe"],
            ["v_0", r["v_0"], "m/s", "Velocity at impeller eye"],

            ["u_1", r["u_1"], "m/s", "Blade velocity at impeller inlet"],
            ["v_1ax", r["v_1ax"], "m/s", "Axial velocity at impeller inlet"],
            ["v_1m", r["v_1m"], "m/s", "Meridional velocity at impeller inlet"],
            ["v_1", r["v_1"], "m/s", "Absolute velocity at impeller inlet"],

            ["u_2", r["u_2"], "m/s", "Blade velocity at impeller outlet"],
            ["v_3", r["v_3"], "m/s", "Velocity in diffuser throat"],
            ["v_4", r["v_4"], "m/s", "Velocity at diffuser outlet"],

            ["T_4", r["T_4"], "K", "Outlet temperature"],
        ]

        print(tabulate(
            rows,
            headers=["Parameter", "Value", "Unit", "Description"],
            tablefmt="github",
            floatfmt=".3f"
        ))