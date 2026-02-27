import copy
import warnings
import numpy as np
import functions
import scipy.optimize as opt
import scipy.interpolate as intrp


class BarskePump:
    def __init__(self):
        """A class for sizing and analysis of Barske pump."""

        # Diameters
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
        self.specific_speed_BEP = None # specific speed of the pump at Best Efficiency Point, EU units (m, RPM, m^3/h)

        #TODO Add forces

        # Analysis results - dictionary to store analysis inputs and results.
        self.analysis_results = {"fluid": None, # Fluid object used for analysis, -
                                 "RPM": None, # Rotations Per Minute, 1 / minute
                                 "w": None,  # Angular speed, rad / s
                                 "mdot": None, # Massflow through pump, kg / s
                                 "dP": None,  # Pressure rise across the pump, Pa
                                 "Q": None, # Volumetric flow through pump (defined with p_0)
                                 "H_useful": None, # Useful head of the pump, m
                                 "H_losses": None,  # Head losses of the pump, m
                                 "H_ideal": None,  # Ideal head of the pump, m
                                 "flow_coefficient_inlet": None, # Flow coefficient at the inlet of the impeller, -
                                 "flow_coefficient_outlet": None, # Flow coefficient in the diffuser throat, -
                                 "head_coefficient": None, # Head coefficient of the pump, -
                                 "P_h_useful": None, # Useful pump power, W
                                 "P_h_losses": None, # Hydraulic power losses, W
                                 "P_h_total": None,  # Total hydraulic power, W
                                 "P_total": None,  # Total pump power, W
                                 "P_f": None,  # Power to overcome friction, W
                                 "eta": None, # Pump efficiency, -
                                 "eta_losses": None, # fraction of dynamic head lost in the diffuser, -
                                 "K_factor": None, # factor for prerotation at zero flow as a fraction of inlet tip speed
                                 # (used in Lock's analysis method), -
                                 "T_upstream": None, # Upstream temperature, K
                                 "p_upstream": None, # Upstream total (tank/reservoir) pressure, Pa
                                 "rho": None, # Fluid density used for the analysis, kg/m^3
                                 "p_0": None, # Pressure at the impeller eye, Pa
                                 "p_1": None, # Pressure at the impeller leading edge, Pa
                                 "p_2": None, # Pressure at the impeller outlet, Pa
                                 "p_3": None, # Pressure in the diffuser throat, Pa
                                 "p_4": None, # Pressure at the diffuser outlet, Pa
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
        solidity = [0.506, 0.606, 0.704, 0.804, 0.904, 1.003, 1.104, 1.202, 1.301, 1.401, 1.501, 1.603, 1.704, 1.803, 1.903,
             2.003, 2.104, 2.206, 2.307, 2.406, 2.505]
        C_h = [0.995, 0.994, 0.988, 0.980, 0.967, 0.955, 0.942, 0.926, 0.908, 0.890, 0.872, 0.854, 0.837, 0.819, 0.803, 0.787,
             0.772, 0.757, 0.743, 0.729, 0.717]
        C_h_interp = intrp.interp1d(solidity, C_h, kind="linear", bounds_error=False, fill_value="extrapolate")
        # Property to store interpolating function
        self.C_h = lambda sigma: np.clip(C_h_interp(sigma), 0.0, 1.0)

    def size_dimensions(self, fluid, RPM, dP, mdot, p_upstream, T_upstream, inlet_sizing_method, diameter_sizing_method,
                        widths_sizing_method, t_hub, t_LE, t_TE, n_blades=5, diffuser_area_ratio = 4,
                        diffuser_angle = 8, flow_coefficient_outlet=0.8, D_1_over_D_0=1.1, D_5_over_D_1 = 1.1,
                        alpha_1 = 90, v_0=3.6576, u_1=45.72, flow_coefficient_inlet=0.07, L_1_over_D_1=0.25,
                        r_factor = 0.8, eta_losses=0.194, K_factor=0.17):
        """A method to size the Barske pump. It updates parameters of the BarskePump object.

        :param fluid: Object representing fluid used for the sizing of the Barske Pump.
        :param float or int RPM: Design Rotations Per Minute at BEP, 1 / minute
        :param float or int dP: Pressure rise across the pump, Pa
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
        :param float or int t_hub: Hub thickness, m
        :param float or int t_LE: Leading edge (suction side) thickness, m
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
        :param float or int eta_losses: Fraction of dynamic head lost in the diffuser. By default, 0.194. This is the
            value used by Lock in "A Forced Vortex Pump for High Speed, High Pressure, Low Flow Applications"
        :param float or int K_factor: Factor for prerotation at zero flow as a fraction of inlet tip speed, used for
            "Lock" option for diameter_sizing_method. By default, 0.17. Middle value from the range used in
             "A Forced Vortex Pump for High Speed, High Pressure, Low Flow Applications".
        """
        # Get fluid density
        rho = fluid.get_density(p_upstream, T_upstream) # kg/s

        # Calculate volumetric flow and the required head. When calculating volumetric flow, it is assumed fluid is
        # incompressible.
        Q_design = mdot / rho  # m^3 / s
        H_required = functions.get_H_from_dP(fluid, dP, p_upstream, T_upstream)

        # Calculate specific speed and assign number of blades
        self.specific_speed_BEP = RPM * np.sqrt(Q_design) / (H_required**0.75)
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
        self.A_4_over_A_3 = diffuser_area_ratio
        self.alpha_diffuser = diffuser_angle # degrees

        #TODO Take thicknesses into account
        #TODO Incorporate splitter blades

        # Find outlet diameter that satisfies requirements
        # First define a function that calculates impeller's dimensions and H as a function of D_2
        def get_impeller_head(D_2):
            # Calculate blade tip speed
            u_2 = D_2 * omega / 2 # m/s

            # Size the diffuser from the assigned outlet flow coefficient
            v_3 = u_2 * flow_coefficient_outlet # m/s
            D_3 = np.sqrt(4 * Q_design / (np.pi * v_3)) # m
            A_3 = np.pi * (D_3 / 2)**2 # m^2
            A_4 = self.A_4_over_A_3 * A_3 # m^2
            D_4 = 2 * np.sqrt(A_4 / np.pi) # m^2
            L_diffuser = ((D_4 - D_3) / 2) / np.tan(self.alpha_diffuser * np.pi / 180) # m

            # If blade width sizing method is Gulich, first get L_2 and then get L_1
            if widths_sizing_method == "Gulich":
                L_2 = (0.02 + 0.5 * self.specific_speed_BEP/100 - 0.03 * (self.specific_speed_BEP/100)**2
                       - 0.04 * (self.specific_speed_BEP/100)**3) * D_2 # m
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
            s_rad = (0.008 * (D_2 / 0.0254) + 0.072) * 0.0254

            # Analyse performance either with Lobanoff or Lock method
            H = ...

            return H, D_3, D_4, A_3, A_4, L_diffuser, L_1, L_2, alpha_0, alpha_2, s_ax, s_rad

        # Now solve the function for D_2 and get all other remaining dimensions
        D_2_estimate = np.sqrt((4 * self.g * H_required) / (0.7 * omega**2))
        self.D_2 = opt.toms748(f=lambda x: H_required - get_impeller_head(x), a=self.D_1 * 1.1, b=3*D_2_estimate)[0]
        (H_design, self.D_3, self.D_4, self.A_3, self.A_4, self.L_diffuser, self.L_1, self.L_2, self.alpha_0,
         self.alpha_2, self.s_ax, self.s_rad) = get_impeller_head(self.D_2)

        # With all dimensions known, full analysis can be performed to get more data about flow and performance
        self.analysis_results_design = self.analyse(...)

    def analysis_Lock(self):
        # First find h_0 and C_h factors from digitalized Figure 10a and Figure 10b from Lock. To do so, ratio of
        # impeller radii and its solidity need to be found.
        h_0 = ...
        C_h = ...
        H = ...
        return H

    def analysis_Lobanoff(self):
        ...


    def assign_dimensions(self):
        """A method to assign pump's dimensions"""
        ...

    def verify_design(self):
        """A method to verify the design of the Barske pump"""
        ...

    def analyse(self):
        """A method to analyze pump's performance for given geometry"""
        # Calculate volumetric flow at the inlet

        # First calculate v_0

        # Then calculate u_1, v_1m and v_1

        # Calculate u_2

        # Calculate paddle power to overcome friction

        # Calculate impeller ideal heads depending on the method

        # Calculate p_2

        # Calculate p_3 and v_3

        # Calculate p_4 and v_4

        # Calculate head loss depending on the method

        # Calculate real heads

        # Calculate useful pump power

        # Calculate pump power lost to hydraulic losses

        # Calculate total hydraulic pump power

        # Calculate total pump power

        # Calculate pump efficiency

        # Calculate pump coefficients

        # Calculate outlet temperature

        # Pack results into dictionary and return it
        analysis_results = ...
        return analysis_results

    def print_dimensions(self):
        """A method to print pump's dimensions"""
        ...

    def plot_geometry(self):
        """A method to plot geometry of the pump"""
        ...

    def print_analysis_results(self):
        """A method to print results of the analysis"""
        ...