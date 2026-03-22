import CoolProp
import CoolProp.CoolProp as cp
import warnings

class Fluid:
    def __init__(self, name, CoolProp_name=None, density=None, dynamic_viscosity=None, vapor_pressure=None,
                 specific_heat=None):
        """A class for modelling fluid properties. It either stores user-given constant properties or uses CoolProp to
         get properties dependent on pressure or temperature.

         :param string name: Name of the fluid.
         :param string CoolProp_name: CoolProp name of the fluid. If it is given, CoolProp will be used to evaluate fluid
            properties. By default, None. If it is not None, and other properties are not None too, CoolProp will be
             used to overwrite them.
         :param float or int density: Density of the fluid, kg/m^3.
         :param float or int dynamic_viscosity: Dynamic viscosity of the fluid, Pa*s.
         :param float or int vapor_pressure: Vapor pressure of the fluid, Pa.
         :param float or int specific_heat: Specific heat of the fluid, J/kg/K.
         """

        # Save name of the fluid
        self.name = name

        # First print an error if all arguments are None
        if all(v is None for v in (CoolProp_name, density, dynamic_viscosity, vapor_pressure)):
            warnings.simplefilter("error", UserWarning)
            warnings.warn("All arguments are None")
        # Print a warning if all argument are given. In such case, CoolProp will be used to overwrite other
        # properties.
        elif all(v is not None for v in (CoolProp_name, density, dynamic_viscosity, vapor_pressure)):
            warnings.warn("All arguments are given. CoolProp will override constant variables.")

        # If name of the CoolProp fluid is provided, create and store map of the fluid for fast interpolation
        if CoolProp_name is not None:
            # Set use_coolprop to flag
            self.use_coolprop = True
            # Create CoolProp table for fast interpolation
            self.coolprop_table = cp.AbstractState("TTSE&HEOS", CoolProp_name)
            # Assing CoolProp name
            self.CoolProp_name = CoolProp_name
        # If name of the CoolProp fluid is not given, use constant properties
        elif CoolProp_name is None:
            # If any of them is None, raise an error
            if all(v is None for v in (density, dynamic_viscosity, vapor_pressure)):
                warnings.simplefilter("error", UserWarning)
                warnings.warn("Inputs 'density', 'dynamic_viscosity', 'vapor_pressure' are all None")
            # Set use_coolprop to flag
            self.use_coolprop = False
            # Assign constant properties
            self.density = density
            self.dynamic_viscosity = dynamic_viscosity
            self.vapor_pressure = vapor_pressure
            self.specific_heat = specific_heat

    def get_density(self, p, T):
        """A method to get density of the fluid.

        :param float or int p: Static pressure of the fluid, Pa
        :param float or int T: Temperature of the fluid, Pa

        :return: Density of the fluid, kg/m^3
        :rtype: float
        """
        if self.use_coolprop:
            self.coolprop_table.update(CoolProp.PT_INPUTS, p, T)
            return self.coolprop_table.rhomass()
        else:
            return self.density

    def get_dynamic_viscosity(self, p, T):
        """A method to get dynamic viscosity of the fluid.

        :param float or int p: Static pressure of the fluid, Pa
        :param float or int T: Temperature of the fluid, Pa

        :return: Dynamic viscosity of the fluid, Pa*s
        :rtype: float
        """
        if self.use_coolprop:
            self.coolprop_table.update(CoolProp.PT_INPUTS, p, T)
            return self.coolprop_table.viscosity()
        else:
            return self.dynamic_viscosity

    def get_kinematic_viscosity(self, p, T):
        """A method to get kinematic viscosity of the fluid.

        :param float or int p: Static pressure of the fluid, Pa
        :param float or int T: Temperature of the fluid, Pa

        :return: Kinematic viscosity of the fluid, m^2/s
        :rtype: float
        """
        if self.use_coolprop:
            self.coolprop_table.update(CoolProp.PT_INPUTS, p, T)
            return self.coolprop_table.viscosity() / self.coolprop_table.rhomass()
        else:
            return self.get_dynamic_viscosity(p, T) / self.get_density(p, T)

    def get_vapor_pressure(self, T):
        """A method to get vapor pressure of the fluid.

        :param float or int p: Static pressure of the fluid, Pa
        :param float or int T: Temperature of the fluid, Pa

        :return: Vapor pressure of the fluid, kg/m^3
        :rtype: float
        """
        if self.use_coolprop:
            self.coolprop_table.update(CoolProp.QT_INPUTS, 1, T)
            return self.coolprop_table.p()
        else:
            return self.vapor_pressure

    def get_specific_heat(self, p, T):
        """A method to get specific heat of the fluid.

        :param float or int p: Static pressure of the fluid, Pa
        :param float or int T: Temperature of the fluid, Pa

        :return: Specific heat of the fluid, J/kg/K
        :rtype: float
        """
        if self.use_coolprop:
            self.coolprop_table.update(CoolProp.PT_INPUTS, p, T)
            return self.coolprop_table.cpmass()
        else:
            return self.specific_heat