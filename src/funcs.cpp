#include "funcs.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

int add(int i, int j)
{
    return i + j;
}

Eigen::VectorXd matrix_mult(py::EigenDRef<Eigen::MatrixXd> mat, Eigen::Ref<Eigen::VectorXd> v)
{
    Eigen::VectorXd out = mat * v;
    return out;
}

/*
Eigen::MatrixXd inv(Eigen::MatrixXd xs)
{
    Eigen::MatrixXd inv_xs = xs.inverse();
    return inv_xs;
}
*/

Eigen::MatrixXd get_matrix()
{
    Eigen::MatrixXd big_mat = Eigen::MatrixXd::Zero(5, 5);
    return big_mat;
}
