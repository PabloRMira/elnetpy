#ifndef FUNCS_H
#define FUNCS_H

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

int add(int i, int j);

Eigen::VectorXd matrix_mult(py::EigenDRef<Eigen::MatrixXd> mat, Eigen::Ref<Eigen::VectorXd> v);

/* Eigen::MatrixXd inv(Eigen::MatrixXd xs); */

Eigen::MatrixXd get_matrix();

#endif
