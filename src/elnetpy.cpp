#include <pybind11/pybind11.h>
#include "funcs.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(elnetpy, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: elnetpy
        .. autosummary::
           :toctree: _generate
           add
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("matrix_mult", &matrix_mult, py::return_value_policy::reference_internal);
    /*    m.def("inv", &inv);*/
    m.def("get_matrix", &get_matrix, py::return_value_policy::reference_internal);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
