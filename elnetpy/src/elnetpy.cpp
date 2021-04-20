#include <pybind11/pybind11.h>
#include "elnet_linear.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(elnetpy, m)
{
    m.doc() = R"pbdoc(
        Elastic Net C++ internal functions
        -----------------------
        .. currentmodule:: elnetpy
        .. autosummary::
           :toctree: _generate
           add
    )pbdoc";

    m.def("elnet_linear", &elnet_linear, py::return_value_policy::reference_internal);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
