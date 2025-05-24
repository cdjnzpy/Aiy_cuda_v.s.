#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <ctime>

using namespace std;

const int N = 7;
const int Nk = 300;

struct Markov_tran
{
    double Trans[N][N];
    double Grid_s[N];
    double prob_dist[N];
};

double get_tolerances(double VA[Nk][N], double VB[Nk][N]) {
    double tolerance = 0;
    for (int i = 0; i < Nk; i++) {
        for (int j = 0; j < N; j++) {
            tolerance += abs(VA[i][j] - VB[i][j]);
        }
    }
    tolerance /= double(Nk) * double(N);
    //cout << "tolerances " << tolerance << endl;
    return tolerance;
}

double cacu_normcdf(double x, double mean, double var)
{
    // constants
    double a1 = 0.254829592;
    double a2 = -0.284496736;
    double a3 = 1.421413741;
    double a4 = -1.453152027;
    double a5 = 1.061405429;
    double p = 0.3275911;

    //do z-transformization
    x = (x - mean) / var;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x) / sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

//tauchen method
Markov_tran markovappr(double rho, double sigma, double m, int N) {
	double stvy = sqrt(pow(sigma, 2.0) / (1.0 - pow(rho, 2.0)));
	double ymax = m * stvy;
	double ymin = -ymax;
	double w = (ymax - ymin) / (N - 1.0);

    struct Markov_tran tran_result;

	//
	double *s = new double[N];

	for (int i = 0; i < N; i++) {
		s[i] = ymin + w * i;
	}

    //2-dim matrix
    double **Trans = new double*[N];
    for (int i = 0; i < N; i++) {
        Trans[i] = new double[N];
    }

    for (int i = 0; i < N; i++) {
        for (int j = 1; j < N - 1; j++) {
            Trans[i][j] = cacu_normcdf(s[j] - rho * s[i] + w / 2.0, 0, sigma) - cacu_normcdf(s[j] - rho * s[i] - w / 2.0, 0, sigma);
        }
        Trans[i][0] = cacu_normcdf(s[0] - rho * s[i] + w / 2.0, 0, sigma);
        Trans[i][N - 1] = 1 - cacu_normcdf(s[N - 1] - rho * s[i] - w / 2.0, 0, sigma);
    }
    //check sum line trans?=1
    for (int i = 0; i < N; i++) {
        double sum_row_trans = 0;
        for (int j = 0; j < N; j++) {
            sum_row_trans += Trans[i][j];
        }
        if (abs(sum_row_trans - 1) > 1e-6) {
            printf("wrong row%d in wrong sum%f\n", i, sum_row_trans);
        }
    }

    double* p = new double[N];
    double* pl = new double[N];

    for (int i = 0; i < N; i++) {
        p[i] = 1.0 / double(N);
    }


    double test = 1;

    //N*N times N*1
    while (test > 1e-8) {
        double max_tol = 0;

        for (int i = 0; i < N; i++) {
            pl[i] = 0;
            for (int j = 0; j < N; j++) {
                pl[i] += Trans[j][i] * p[j];
            }
            double tol = abs(pl[i] - p[i]);
            if (tol > max_tol) {
                max_tol = tol;
            }
            
        }

        test = max_tol;

        for (int i = 0; i < N; i++) {
            p[i] = pl[i];
        }

    }

    //output save
    for (int i = 0; i < N; i++) {
        tran_result.Grid_s[i] = s[i];
        tran_result.prob_dist[i] = p[i];
        for (int j = 0; j < N; j++) {
            
            tran_result.Trans[i][j] = Trans[i][j];
        }
    }

    //free memory
    delete[] p;
    delete[] pl;
    delete[] s;
    for (int i = 0; i < N; i++) {
        delete[] Trans[i];
    }
    delete[] Trans;

    return tran_result;
}


//dim matrix

double logs[N];
double Trans[N][N];
double inv_dist[N];

double k_list[Nk];
double c_list[Nk * Nk][N], u_list[Nk * Nk][N];

double V0[Nk][N];
double V1[Nk][N];
int policy_k[Nk][N];

double trans_matrix[Nk * N][Nk * N] = { 0 };

double probst[Nk * N];
double probst1[Nk * N] = { 0 };

void main() {
    double mu = 3;
    double sigma = 0.2;
    double rho = 0.6;

    double beta = 0.96;
    double theta = 0.36;
    double delta = 0.08;



    double sigmain = sigma * sqrt(1 - pow(rho, 2));

    Markov_tran markovappr(double rho, double sigma, double m, int N);

    Markov_tran tran_result = markovappr(rho, sigmain, 3, N);



    for (int i = 0; i < N; i++) {
        logs[i] = tran_result.Grid_s[i];
        inv_dist[i] = tran_result.prob_dist[i];
        for (int j = 0; j < N; j++) {
            Trans[i][j] = tran_result.Trans[i][j];
        }
    }

    double shock_s[N];
    double labor = 0;

    for (int i = 0; i < N; i++) {
        shock_s[i] = exp(logs[i]);
        labor += inv_dist[i] * shock_s[i];
    }

    //dim tolerances
    double tolv = 1e-7;
    double tolr = 1e-4;
    double tola = 5e-3;

    double r = 0.03;

    //cacu k
    double phi = 0;

    double ks = pow(delta, 1.0 / (theta - 1.0)) * labor;

    double min_kap = -phi;
    double max_kap = 0.5 * ks;

    double each_k_step = (max_kap - min_kap) / (Nk-1.0);



    for (int i = 0; i < Nk; i++) {
        k_list[i] = min_kap + double(i) * each_k_step;
    }

    double r_h = 0;
    double r_l = 0;

    double starttime, endtime;

    //general r eq
    int iter1 = 1;
    while (iter1 <= 50) {
        starttime = clock();
        double k = pow((r + delta) / (theta * pow(labor, 1.0 - theta)), 1.0 / (theta - 1.0));
        double w = (1.0 - theta) * pow(k, theta) * pow(labor, -theta);

        
        for (int i_shock = 0; i_shock < N; i_shock++) {
            for (int i_k = 0; i_k < Nk; i_k++) {
                for (int i_kk = 0; i_kk < Nk; i_kk++) {
                    c_list[i_k * Nk + i_kk][i_shock] = (1 + r) * k_list[i_k] + w * shock_s[i_shock] - k_list[i_kk];
                    if (c_list[i_k * Nk + i_kk][i_shock] < 0) {
                        c_list[i_k * Nk + i_kk][i_shock] = 1e-7;
                    }
                    u_list[i_k * Nk + i_kk][i_shock] = (pow(c_list[i_k * Nk + i_kk][i_shock], 1 - mu) - 1) / (1 - mu);
                }
            }
        }

        endtime = clock();

        for (int i = 0; i < Nk; i++) {
            for (int j = 0; j < N; j++) {
                V0[i][j] = 1;
            }
        }

        int count = 0;
        while (get_tolerances(V0, V1) > tolv) {
            count += 1;
            //cout << "iter times " << count << endl;
            for (int i = 0; i < Nk; i++) {
                for (int k = 0; k < N; k++) {
                    V0[i][k] = V1[i][k];
                }
            }

            for (int k = 0; k < N; k++) {
                for (int i = 0; i < Nk; i++) {

                    //cacu utility

                    double V_line[Nk] = { 0 };

                    for (int j = 0; j < Nk; j++) {
                        for (int kk = 0; kk < N; kk++) {
                            V_line[j] += beta * V0[j][kk] * Trans[k][kk];
                        }
                        V_line[j] += u_list[i * Nk + j][k];
                    }
                    //find max and index
                    int max_index = 0;
                    double max_value = -1e10;

                    for (int j = 0; j < Nk; j++) {

                        if (V_line[j] > max_value) {
                            max_value = V_line[j];

                            max_index = j;

                            //cout << "out data " << max_index<<" iter" << j << endl;
                        }
                    }

                    V1[i][k] = max_value;
                    policy_k[i][k] = max_index;

                }
            }
        }

        //cacu steady dist
        for (int i = 0; i < Nk; i++) {
            for (int k = 0; k < N; k++) {
                for (int ii = 0; ii < Nk; ii++) {
                    for (int kk = 0; kk < N; kk++) {
                        trans_matrix[k * Nk + i][Nk * kk + ii] = 0;
                    }
                }
            }
        }
        

        for (int i = 0; i < Nk; i++) {
            for (int k = 0; k < N; k++) {
                for (int kk = 0; kk < N; kk++) {
                    trans_matrix[k * Nk + i][Nk * kk + policy_k[i][k]] += Trans[k][kk];
                }
            }
        }



        for (int i = 0; i < Nk; i++) {
            for (int k = 0; k < N; k++) {
                probst[k * Nk + i] = 1.0 / (double(Nk) * double(N));
            }
        }

        
        double test = 1.0;

        while (test > 1e-5) {
            printf("now in test%f\n", test);
            double max_tol = 0;
            //Nk*N * Nk*N times Nk*N * 1
            for (int i = 0; i < Nk; i++) {
                for (int k = 0; k < N; k++) {
                    probst1[k * Nk + i] = 0;

                    for (int ii = 0; ii < Nk; ii++) {
                        for (int kk = 0; kk < N; kk++) {
                            probst1[k * Nk + i] += trans_matrix[kk * Nk + ii][k * Nk + i] * probst[kk * Nk + ii];
                        }
                    }
                    double tol = abs(probst[k * Nk + i] - probst1[k * Nk + i]);
                    if (tol > max_tol) {
                        max_tol = tol;
                    }
                }
            }
            test = max_tol;

            for (int i = 0; i < Nk; i++) {
                for (int k = 0; k < N; k++) {
                    probst[k * Nk + i] = probst1[k * Nk + i];
                    //printf(" %.10f ", probst1[k * Nk + i]);
                }
                //printf("\n");
            }            
        }

        double mean_k = 0;
        for (int i = 0; i < Nk; i++) {
            for (int k = 0; k < N; k++) {
                mean_k += probst[k * Nk + i]*k_list[policy_k[i][k]];
            }
        }

        double r_star = theta * (pow(mean_k, theta - 1) * pow(labor, 1 - theta)) - delta;

        double tol_this_k = abs(k - mean_k);

        if (tol_this_k < tola) {
            break;
        }

        if (iter1 == 1) {
            r_star = 1.0 / beta - 1.0;
            if (r >= r_star) {
                r_h = r;
                r_l = r_star;
            }
            else {
                r_h = r_star;
                r_l = r;
            }
        }
        else if (mean_k > k) {
            r_h = r;
        }  
        else {
            r_l = r;
        }
        printf("now end iter%d for r in %f for meank k : %.10f %.10f \n", iter1, r,mean_k,k);

        double tol_this_r = abs(r - (0.5 * r_h + 0.5 * r_l));
        if (tol_this_r < tolr) {
            //break;
        }

        r = 0.5 * r_h + 0.5 * r_l;
        iter1 += 1;

        printf("new r %f\n", r);


    }

}









