from tfdiffeq.odeint import odeint
# from tfdiffeq.adjoint import odeint_adjoint

from tfdiffeq.misc import cast_double, func_cast_double
from tfdiffeq.misc import move_to_device
from tfdiffeq.viz_utils import (plot_phase_portrait,
                                plot_vector_field,
                                plot_results)


__all__ = ['odeint',
           'cast_double', 'func_cast_double', 'move_to_device',
           'plot_phase_portrait', 'plot_vector_field', 'plot_results',
           ]


__version__ = '0.0.1'
