"""
Examples for the CellML 2019 Workshop slides.

Author:
    Peter Cudmore (peter.cudmore@uqconnect.edu.au)

Date:
    03/05/2019


"""

import BondGraphTools as bgt
import sympy as sp
from math import pi
from collections import namedtuple

Vessel = namedtuple("Vessel", ["length", "radius", "thickness", "youngs_modulus"])
"""Wrapper for vessel material parameters"""

Fluid = namedtuple("Fluid", ["density", "viscosity"])
"""Wrapper for fluid material parameters"""


def print_tree(bond_graph, prefix=""):
    """Prints out the hierarchy of the given bond graph"""
    print(f"{prefix}{bond_graph}")
    try:
        for component in reversed(bond_graph.components):
            if prefix == "":
                print_tree(component, prefix +"|-" )
            else:
                print_tree(component, prefix +"-" )
    except AttributeError:
        pass


def filter_factory(name='Example Bond Graph'):
    model = bgt.new(name=name)
    r, l, c = sp.symbols('r, l, c')

    R = bgt.new('R', value=r, name='R')
    L = bgt.new('I', value=l, name='L')
    C = bgt.new('C', value=c, name='C')

    common_flow = bgt.new('1')
    port = bgt.new('SS', name='port')

    components = [R, L, C, common_flow, port]
    bonds = [
        (port, common_flow),
        (common_flow, R),
        (common_flow, L),
        (common_flow, C)
    ]

    bgt.add(model, components)

    for tail, head in bonds:
        bgt.connect(tail, head)

    bgt.expose(port)

    return model


class VesselSegmentA(bgt.BondGraph):
    """A vascular vessel segment.

    This class is an example of Vessel Segment A from:
    Safaei, Sorous. Blanco, Pablo J. Müller, Lucas O. Hellevik, Leif R. and Hunter, Peter J.
    Bond Graph Model of Cerebral Circulation: Toward Clinically Feasible Systemic Blood Flow Simulations
    Frontiers in Physiology, 2018, volume 9, page 148

    The vessel segment is of the $uv$ type that has components:
        - pressure inlet $u_i$
        - flow outlet $v_o$
        - fluid interia $I$
        - wall dissipation $R$
        - and compliance $C$.

    The linear resistance, compliance and inertance are computed as per Safaei et.al..

    See Also: BondGraph
    """

    def __init__(self, name, vessel, fluid):
        """
        Args:
            name (str):      The name of this vessel segement
            vessel (Vessel): The vessel material properties
            fluid (Fluid):   The fluid properties
        """
        # Parameters
        resistance = 8 * fluid.viscosity * vessel.length / (pi * vessel.radius**4)
        inertance = 2 * pi * vessel.radius**3 / (vessel.thickness * vessel.youngs_modulus)
        compliance = fluid.density * vessel.length / (pi* vessel.radius**2)

        # Instantiating Components
        R_component = bgt.new('R', value=resistance, name=f'R')
        C_component = bgt.new('C', value=compliance, name=f'C')
        I_component = bgt.new('I', value=inertance, name=f'I')

        conserved_flow = bgt.new("1", name=f'1')
        conserved_pressure = bgt.new("0", name=f'0')

        u_in = bgt.new('SS', name=f'u_i')
        v_out = bgt.new('SS', name=f'v_o')

        bonds = [
            (u_in, conserved_flow),
            (conserved_flow, R_component),
            (conserved_flow, I_component),
            (conserved_flow, conserved_pressure),
            (conserved_pressure, C_component),
            (conserved_pressure, v_out)
        ]

        # Build the BondGraph via the inherited initialise function
        super().__init__(
            name=name,
            components=(R_component, C_component, I_component, conserved_flow, conserved_pressure, u_in, v_out)
        )

        # wire it up
        for bond in bonds:
            bgt.connect(*bond)

        # expose the ports
        bgt.expose(u_in, label="u_i")
        bgt.expose(v_out, label="v_o")
