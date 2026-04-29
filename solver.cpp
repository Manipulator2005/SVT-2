#include "inmost.h"
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iomanip>

using namespace INMOST;
using namespace std;

namespace functions
{
    double exact_solution(double x, double y)
    {
        return std::sin(4 * x) * std::cos(3 * y);
    }

    double get_f(double x, double y)
    {
        return 25 * std::sin(4 * x) * std::cos(3 * y);
    }

    double upper_bound(double x) { return exact_solution(x, 1.0); }
    double lower_bound(double x) { return exact_solution(x, 0.0); }
    double left_bound(double y) { return exact_solution(0.0, y); }
    double right_bound(double y) { return exact_solution(1.0, y); }
}

int main(int argc, char *argv[])
{
    Solver::Initialize(&argc, &argv);

    std::vector<unsigned> n_values = {10, 20, 40, 80, 160, 320, 640, 1280};
    // Output columns of the \"answer table\"
    std::cout << std::setw(8) << "h"
              << std::setw(16) << "C-norm error"
              << std::setw(16) << "L2-norm error"
              << std::setw(12) << "Iters"
              << std::setw(12) << "Iter.time"
              << std::endl;

    // Get current n value
    for (unsigned n : n_values)
    {
        unsigned N = n * n;
        double h = 1.0 / (n + 1);

        Sparse::Matrix A;
        Sparse::Vector b;
        Sparse::Vector sol;
        A.SetInterval(0, N);
        b.SetInterval(0, N);
        sol.SetInterval(0, N);
	
	// Put values into A matrix
        for (unsigned i = 0; i < N; ++i)
        {
            A[i][i] = 4.0;

            int row = i / n;
            int col = i % n;
            if (col > 0)
                A[i][i - 1] = -1.0;
            if (col < n - 1)
                A[i][i + 1] = -1.0;
            if (row > 0)
                A[i][i - n] = -1.0;
            if (row < n - 1)
                A[i][i + n] = -1.0;
        }
	// Put values into b vector
        for (unsigned i = 0; i < n; ++i)
        {
            double y = (i + 1) * h;
            for (unsigned j = 0; j < n; ++j)
            {
                double x = (j + 1) * h;
                unsigned idx = i * n + j;

                b[idx] = functions::get_f(x, y) * h * h;

                if (i == 0)
                    b[idx] += functions::lower_bound(x);
                if (i == n - 1)
                    b[idx] += functions::upper_bound(x);
                if (j == 0)
                    b[idx] += functions::left_bound(y);
                if (j == n - 1)
                    b[idx] += functions::right_bound(y);
            }
        }

        Solver S(Solver::INNER_ILU2);
        S.SetParameter("absolute_tolerance", "1e-14");
        S.SetParameter("relative_tolerance", "1e-11");
        S.SetMatrix(A);

        bool solved = S.Solve(b, sol);
        if (!solved)
        {
            std::cout << "Solver failed for n = " << n << std::endl;
            continue;
        }

        double C_error = 0.0;
        double L2_error = 0.0;

        for (unsigned i = 0; i <= n + 1; ++i)
        {
            double y = i * h;
            for (unsigned j = 0; j <= n + 1; ++j)
            {
                double x = j * h;
                double exact = functions::exact_solution(x, y);
                double approx;

                if (i == 0)
                    approx = functions::lower_bound(x);
                else if (i == n + 1)
                    approx = functions::upper_bound(x);
                else if (j == 0)
                    approx = functions::left_bound(y);
                else if (j == n + 1)
                    approx = functions::right_bound(y);
                else
                {
                    unsigned idx = (i - 1) * n + (j - 1);
                    approx = sol[idx];
                }

                double diff = std::abs(approx - exact);
                if (diff > C_error)
                    C_error = diff;
                L2_error += diff * diff;
            }
        }
        L2_error = std::sqrt(L2_error * h * h);

        std::cout << std::setw(8) << h
                  << std::setw(16) << std::scientific << C_error
                  << std::setw(16) << L2_error
                  << std::setw(8) << S.Iterations()
                  << std::setw(16) << S.IterationsTime()
                  << std::endl;
    }

    Solver::Finalize();
    return 0;
}
