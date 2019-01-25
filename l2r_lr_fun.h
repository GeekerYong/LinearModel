//
// Created by 门捷列夫 on 2019-01-23.
//

#ifndef LINEARMODEL_L2R_LR_FUN_H
#define LINEARMODEL_L2R_LR_FUN_H

#include "function.h"

class l2r_lr_fun : public function {
public:
	l2r_lr_fun(const problem *prob, double *C);

	~l2r_lr_fun();

	double fun(double *w);

	virtual void grad(double *w, double *g);

	virtual void Hv(double *s, double *Hs);

	virtual int get_nr_variable(void);

private:
	void Xv(double *v, double * Xv);
	void XTv(double* v, double * XTv);
	double * C; //惩罚因子，可以为每个样本设定一个独立的惩罚
	double * z; //临时变量，用于存储wTx
	double * D; //临时变量，存储logistic函数的导数，来计算Hessian矩阵
	const problem * prob;
};


#endif //LINEARMODEL_L2R_LR_FUN_H
