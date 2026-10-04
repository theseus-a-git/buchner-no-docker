# -*- coding: utf-8 -*-
"""
@file dispatch_tables.py

@brief Dispatch tables used throughout the PSO package.

@author
Ben Carlson

@date
2026-08-28
"""
from options.boundary_types.let_them_fly import let_them_fly
from options.boundary_types.absorbing_wall import absorbing_wall
from options.boundary_types.reflecting_wall import reflecting_wall
from options.boundary_types.periodic import periodic
from options.boundary_types.custom import custom

from options.boundary_types.ltf_23triangle import LTF_23Triangle
from options.boundary_types.ltf_23triangle_refl import LTF_23Triangle_Refl
from options.boundary_types.imrpd_1 import IMRPD_1

BOUNDARY_HANDLERS = {
        "let_them_fly": let_them_fly,
        "absorbing_wall": absorbing_wall,
        "reflecting_wall": reflecting_wall,
        "periodic": periodic,
        "custom": custom,
        "ltf_23triangle": LTF_23Triangle,
        "ltf_23triangle_refl": LTF_23Triangle_Refl,
        "IMRPD-1": IMRPD_1,
    }

SUPPORTED_BOUNDARY_TYPES = frozenset(BOUNDARY_HANDLERS)

###############################################################################
###############################################################################
###############################################################################

from options.value_generators.halton import make_halton
from options.value_generators.random_uniform import make_random_uniform
from options.value_generators.sobol import make_sobol
from options.value_generators.premade import make_premade

VALUE_GENERATORS = {
        "halton": make_halton,
        "random-uniform": make_random_uniform,
        "sobol": make_sobol,
        "premade": make_premade,
    }

SUPPORTED_VALUE_GENERATORS = frozenset(VALUE_GENERATORS)

###############################################################################
###############################################################################
###############################################################################

