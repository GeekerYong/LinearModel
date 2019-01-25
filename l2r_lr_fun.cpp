//
// Created by 门捷列夫 on 2019-01-23.
//

#include <cmath>
#include "l2r_lr_fun.h"


double l2r_lr_fun::fun(double *w) {
    int i;
    double f = 0; //损失函数的值
    double *y = prob->y; //标签
    int l = prob->l; //样本的数目
    int w_size = get_nr_variable(); //特征的数目
    //正则项的损失
    for (int i = 0; i < w_size; ++i) {
        f += w[i] * w[i];
    }
    f /= 2.0;
    //常规项的损失
    for (int j = 0; j < l; ++j) {
        double yz = y[j] * z[j];
        if (yz >= 0) {
            f += C[i] * log(1 + exp(-yz));
        } else {
            f += C[i] * (-yz + log(1 + exp(yz))); //这个式子的意义需要探究
        };
    }
    return f;
}

void l2r_lr_fun::grad(double *w, double *g) {
    double *y = prob->y;
    int l = prob->l;
    int w_size = get_nr_variable();
    for (int i = 0; i < w_size; ++i) {
        z[i] = 1/(1 + exp(-y[i]*z[i]));
        D[i] = z[i]*(1-z[i]);
        z[i] = C[i] *(z[i] -1)*y[i];
    }
    XTv(z, g);
    for (int j = 0; j < l; ++j) {
        g[j] = w[j] + g[j];
    }
}

void l2r_lr_fun::Hv(double *s , double * Hs){
    int l = prob->l;
    int w_size = get_nr_variable();
    double *wa = new double[];
    Xv(s, wa);
    for (int i = 0; i < l; ++i) {
        wa[i] = C[i]*D[i]*wa[i];
    }
}
