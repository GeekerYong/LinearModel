//
// Created by 门捷列夫 on 2019-01-23.
//

#ifndef LINEARMODEL_FUNCTION_H
#define LINEARMODEL_FUNCTION_H

struct feature_node {
	int index; //特征编号
	double value; //特征的值
};

struct problem {
	int l, n; //样本数，特征长度
	double *y; //标签值 --分类：类别号-- ，--回归：实数值--
	struct feature_node **x; //数据集，每行为一个样本
	double bias; //偏置值b
};
struct parameter {
	int solver_type;//求解器
	double eps;//算法迭代停止的阈值
	double C;//惩罚因子
	int nr_weight;//权重数组的大小
	int *weight_label;//权重标签数组
	double *weight;//权重数组
	double p;
};

struct model{
	struct parameter param; //训练参数
	int nr_class; //类别数量
	int nr_feature; //特征数量
	double * w; //权重向量
	int * label; //每个类的类别标签
	double bias; //偏置项b
};

class function {
//损失函数的基类
//功能：
// （1）计算损失函数值
// （2）计算梯度值
// （3）计算Hessian矩阵

public:
	//计算目标函数的值，w为权重数组，返回的是损失函数的值
	virtual double fun(double *w) =0 ;
	//计算目标函数的梯度，w为权重数组，g为返回的梯度值
	virtual void grad(double *w, double *g) =0;
	//计算目标函数Hessian矩阵与传入向量的乘积，s为传入向量，Hs为返回的结果
	virtual void Hv(double *s, double *Hs) =0;
	virtual int get_nr_variable(void) =0;//获取特征向量的维数
	virtual ~function(void){}
};


#endif //LINEARMODEL_FUNCTION_H
