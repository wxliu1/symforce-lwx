from symengine.lib.symengine_wrapper import ccode, sympify, Basic
from symengine.lib import symengine_wrapper

class CCodePrinter:

    def doprint(self, expr, assign_to=None):
        if not isinstance(assign_to, (Basic, type(None), str)):
            raise TypeError("{0} cannot assign to object of type {1}".format(
                    type(self).__name__, type(assign_to)))

        expr = sympify(expr)
        if not assign_to:
            if expr.is_Matrix:
                raise RuntimeError("Matrices need a assign_to parameter")
            return ccode(expr)

        assign_to = str(assign_to)
        if not expr.is_Matrix:
            return "{} = {};".format(assign_to, ccode(expr))

        code_lines = []
        for i, element in enumerate(expr):
            code_line = '{}[{}] = {};'.format(assign_to, i, element)
            code_lines.append(code_line)
        return '\n'.join(code_lines)


def init_printing(pretty_print=True, use_latex=True):
    if pretty_print:
        if not use_latex:
            raise RuntimeError("Only latex is supported for pretty printing")
        symengine_wrapper.repr_latex[0] = True
    else:
        symengine_wrapper.repr_latex[0] = False
