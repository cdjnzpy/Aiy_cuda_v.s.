#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include "sm_60_atomic_functions.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <ctime>

#include "./book.h"


using namespace std;

//para meters set
double mu = 3.0;
double sigma = 0.2;
double rho = 0.6;

double beta = 0.96;
double theta = 0.36;
double delta = 0.08;

double tolv = 1e-8;
double tolr = 1e-4;
double tola = 5e-3;

double r_h, r_l;
double w, k, r;

//Set grid parameters
const int N = 7;
const int Nk = 3000;

//Set grid list
double logs[N];
double Trans[N][N];
double inv_dist[N];

double k_list[Nk];
double c_list[Nk * Nk][N], u_list[Nk * Nk * N];

double trans_matrix[Nk * N][Nk * N] = { 0 };

double probst[Nk * N];
double probst1[Nk * N];
double probst1_N[Nk * N * N];
double shock_s[N];

double trans_list[N * N];

//in solving 
double V0[N * Nk], V1[N * Nk];
int policy_k[Nk * N];





__constant__ double k_list_constant[Nk];
__constant__ double s_shock_constant[N];
__constant__ double params_input[3];
__constant__ double Trans_constant[N * N];

__global__ void back_utility( double* u_list) {
    //(7,) * 300
    int i_s = blockIdx.x / 10;
    int i_x_now_k = (blockIdx.x - 10 * i_s) * blockDim.x + threadIdx.x;

    double r_cuda = params_input[0];
    double w_cuda = params_input[1];
    double mu_cuda = params_input[2];
    
    

    for (int i_x_next_k = 0; i_x_next_k < Nk; i_x_next_k++) {
        double c = max(1e-7, (1 + r_cuda) * k_list_constant[i_x_now_k] + w_cuda * s_shock_constant[i_s] - k_list_constant[i_x_next_k]);

        double u = (pow(c, 1 - mu_cuda) - 1) / (1 - mu_cuda);

        u_list[i_s * (Nk * Nk) + i_x_now_k * Nk + i_x_next_k] = u;
    }

    
}

__global__ void back_max_value(double* V_input, double* u_list_output, double *V_output, int *policy_choice) {
    // 7*300 
    // 70*30
    // 55 25 5 150+25
    int i_s = blockIdx.x / 10;
    int i_k = (blockIdx.x - 10 * i_s) * blockDim.x + threadIdx.x;

    double V_line[Nk] = { 0 };


    for (int i_kk = 0; i_kk < Nk; i_kk++) {
        for (int i_ss = 0; i_ss < N; i_ss++) {

            V_line[i_kk] += 0.96 * V_input[i_ss * Nk + i_kk] * Trans_constant[i_s * N + i_ss];

        }

        V_line[i_kk] += u_list_output[i_s * Nk * Nk + i_k * Nk + i_kk];
    }


    //find max and index
    int max_index = 0;
    double max_value = -1e21;

    for (int i = 0; i < Nk; i++) {

        if (V_line[i] > max_value) {
            max_value = V_line[i];
            max_index = i;
        }
    }




    //printf("check V1; %.5f %.5f %.5f \n", V1[25], V1[125], V1[625]);
    V_output[i_s * Nk + i_k] = max_value;
    policy_choice[i_s * Nk + i_k] = max_index;

}

__global__ void cacu_dist_steady(double* probst_input, double* probst_output_N) {
    //70,7 * 30
    int i_s = blockIdx.x / 10;
    int i_ss = blockIdx.y;
    int i_k = (blockIdx.x - 10 * i_s) * blockDim.x + threadIdx.x;

    int now_index = i_s * Nk + i_k;

    probst_output_N[i_s * (N * Nk) + i_ss * Nk + i_k] = probst_input[now_index] * Trans_constant[i_s * N + i_ss];


}

double get_tolerances(double VA[Nk*N], double VB[Nk*N]) {
    double tolerance = 0;
    for (int i = 0; i < Nk; i++) {
        for (int j = 0; j < N; j++) {
            tolerance += abs(VA[j*Nk+i] - VB[j*Nk+i]);
        }
    }
    tolerance /= double(Nk) * double(N);
    //cout << "tolerances " << tolerance << endl;
    return tolerance;
}


struct Markov_tran
{
    double Trans[N][N];
    double Grid_s[N];
    double prob_dist[N];
};

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
    double* s = new double[N];

    for (int i = 0; i < N; i++) {
        s[i] = ymin + w * i;
    }

    //2-dim matrix
    double** Trans = new double* [N];
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


void main() {
    double sigmain = sigma * sqrt(1 - pow(rho, 2));
    Markov_tran markovappr(double rho, double sigma, double m, int N);

    Markov_tran tran_result = markovappr(rho, sigmain, 3, N);

    for (int i = 0; i < N; i++) {
        logs[i] = tran_result.Grid_s[i];
        inv_dist[i] = tran_result.prob_dist[i];
        for (int j = 0; j < N; j++) {
            Trans[i][j] = tran_result.Trans[i][j];
            trans_list[i * N + j] = tran_result.Trans[i][j];
        }
    }

    double labor = 0;

    for (int i = 0; i < N; i++) {
        shock_s[i] = exp(logs[i]);
        labor += inv_dist[i] * shock_s[i];
    }

    r = 0.03;
    double phi = 0.0;

    double ks = pow(delta, 1.0 / (theta - 1.0)) * labor;

    double min_kap = -phi;
    double max_kap = 0.5 * ks;

    double each_k_step = (max_kap - min_kap) / (Nk - 1.0);

    for (int i = 0; i < Nk; i++) {
        k_list[i] = min_kap + double(i) * each_k_step;
    }
    double starttime, endtime;
    int iter1 = 1;

    //allocate __Constant__
    
    cudaMemcpyToSymbol(k_list_constant, k_list, sizeof(double) * Nk);
    cudaMemcpyToSymbol(s_shock_constant, shock_s, sizeof(double) * N);
    
    cudaMemcpyToSymbol(Trans_constant, trans_list, N * N * sizeof(double));

    starttime = clock();
    while (iter1 <= 50) {
        
        k = pow((r + delta) / (theta * pow(labor, 1.0 - theta)), 1.0 / (theta - 1.0));
        w = (1.0 - theta) * pow(k, theta) * pow(labor, -theta);
        double para_l_cpu[3] = { r,w,mu };
        cudaMemcpyToSymbol(params_input, para_l_cpu, sizeof(double) * 3);

        //get utility
        double* k_list_input;
        double* u_list_output;
        double* shock_input;

        
        HANDLE_ERROR(cudaMalloc((void**)&u_list_output, N * Nk * Nk * sizeof(double)));

        //dim thread and blocks
        
        back_utility <<< N*10, Nk/10 >>> (u_list_output);

        HANDLE_ERROR(cudaMemcpy(u_list, u_list_output, N * Nk * Nk * sizeof(double), cudaMemcpyDeviceToHost));

        

        cudaFree(k_list_constant);
        cudaFree(s_shock_constant);
        cudaFree(params_input);

        for (int i = 0; i < Nk; i++) {
            for (int j = 0; j < N; j++) {
                V0[j * Nk + i] = 1.0;
                V1[j * Nk + i] = 0.0;
            }
        }

        int count = 0;

        double* V_input;
        double* V_output;
        int* policy_choice;

        
        HANDLE_ERROR(cudaMalloc((void**)&V_input, Nk * N * sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void**)&V_output, Nk * N * sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void**)&policy_choice, Nk * N * sizeof(int)));


        //*************************************************************************************************************

        while (get_tolerances(V0, V1) > tolv) {
            count += 1;

            for (int i = 0; i < Nk; i++) {
                for (int s = 0; s < N; s++) {
                    V0[s * Nk + i] = V1[s * Nk + i];
                }
            }
            //solve
            HANDLE_ERROR(cudaMemcpy(V_input, V0, Nk * N * sizeof(double), cudaMemcpyHostToDevice));

            back_max_value <<< N * 10, Nk / 10 >>> (V_input, u_list_output, V_output, policy_choice);

            HANDLE_ERROR(cudaMemcpy(V1, V_output, Nk * N * sizeof(double), cudaMemcpyDeviceToHost));
        }

        HANDLE_ERROR(cudaMemcpy(policy_k, policy_choice, Nk* N * sizeof(int), cudaMemcpyDeviceToHost));


        cudaFree(V_input);
        cudaFree(V_output);
        cudaFree(policy_choice);
        
        //*************************************************************************************************************


        //draft
        for (int i = 0; i < Nk; i++) {
            for (int s = 0; s < N; s++) {
                probst[s * Nk + i] = 1.0 / (double(Nk) * double(N));
            }
        }

        //Solve steady state
        //need probst1

        double* probst_input;
        double* probst_output_N;


        HANDLE_ERROR(cudaMalloc((void**)&probst_input, Nk * N * sizeof(double)));
        HANDLE_ERROR(cudaMalloc((void**)&probst_output_N, Nk * N * N * sizeof(double)));


        double test_probst = 1.0;
        while (test_probst > 1e-5) {
            //printf("show testprobst %.8f \n", test_probst);
            
            HANDLE_ERROR(cudaMemcpy(probst_input, probst, Nk * N * sizeof(double), cudaMemcpyHostToDevice));


            dim3 probst_block(N * 10, N);
            cacu_dist_steady <<<probst_block, Nk / 10 >>> (probst_input, probst_output_N);

            HANDLE_ERROR(cudaMemcpy(probst1_N, probst_output_N, Nk * N * N * sizeof(double), cudaMemcpyDeviceToHost));

            
  

            for (int i_k = 0; i_k < Nk; i_k++) {
                for (int i_s = 0; i_s < N; i_s++) {
                    for (int i_ss = 0; i_ss < N; i_ss++) {
                        probst1[i_ss * Nk + policy_k[i_s * Nk + i_k]] += probst1_N[i_s * (N * Nk) + i_ss * Nk + i_k];
                    }
                }
            }

            int max_index = 0;
            double max_value_probst = -1.0;

            for (int i = 0; i < N * Nk; i++) {

                if (abs(probst1[i] - probst[i]) > max_value_probst) {
                    max_value_probst = abs(probst1[i] - probst[i]);

                    max_index = i;
                }

                probst[i] = probst1[i];
                probst1[i] = 0.0;
            }
            test_probst = max_value_probst;
        }

        cudaFree(u_list_output);
        cudaFree(probst_input);

        
        double mean_k = 0;
        for (int i_k = 0; i_k < Nk; i_k++) {
            for (int i_s = 0; i_s < N; i_s++) {
                mean_k += probst[i_s * Nk + i_k] * k_list[policy_k[i_s * Nk + i_k]];
                
            }
        }

        double tol_this_k = abs(k - mean_k);
        double r_star = theta * (pow(mean_k, theta - 1) * pow(labor, 1 - theta)) - delta;
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
        printf("now end iter%d for r in %f for meank k : %.10f %.10f \n", iter1, r, mean_k, k);

        double tol_this_r = abs(r - (0.5 * r_h + 0.5 * r_l));

        r = 0.5 * r_h + 0.5 * r_l;
        iter1 += 1;

        printf("new r %f\n", r);
        if (iter1 == 10) {
            break;
        }
    }
    double end_time = clock();

    printf("run time in %.10f \n", (end_time - starttime) / 1000);

}