#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}

/**
 * 标定外参的旋转矩阵
 * @param corres            匹配的特征点
 * @param delta_q_imu       IMU预积分得的旋转矩阵Q
 * @param calib_ric_result  标定的外参数
 * @return
 */
// 标定imu和相机之间的旋转外参，通过imu和图像计算的旋转使用手眼标定计算获得
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    frame_count++;
    // 根据特征关联求解两个连续帧相机的旋转R12
    Rc.push_back(solveRelativeR(corres));   // 帧间cam的R，由对极几何得到
    Rimu.push_back(delta_q_imu.toRotationMatrix());     // 帧间IMU的R，由IMU预积分得到
    // 通过外参把imu的旋转转移到相机坐标系
    // 每帧IMU相对于起始帧IMU的R,初始化为IMU预积分
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);  // ric是上一次求解得到的外参

    // SVD分解，Ax=0,对A填充。 多帧组成A，一对点组成的A是4×4的
    Eigen::MatrixXd A(frame_count * 4, 4);
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++)
    {
        Quaterniond r1(Rc[i]);  // 相机的旋转
        Quaterniond r2(Rc_g[i]);    // IMU转到相机坐标系下的旋转

        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        ROS_DEBUG(
            "%d %f", i, angular_distance);
        // 一个简单的核函数
        // 如果相差的角度值大于5度，不是很合理，就将huber核设为5/角度
        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        Matrix4d L, R;

        //R_bk+1^bk * R_c^b = R_c^b * R_ck+1^ck
        //[Q1(q_bk+1^bk) - Q2(q_ck+1^ck)] * q_c^b = 0
        //L R 分别为左乘和右乘矩阵，都是4×4矩阵
        double w = Quaterniond(Rc[i]).w();      // 相机间旋转矩阵
        Vector3d q = Quaterniond(Rc[i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(Rimu[i]);       // imu间旋转矩阵
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);    // 作用在残差上面
    }

    // svd分解中最小奇异值对应的右奇异向量作为旋转四元数
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    // 这里的四元数存储的顺序是[x，y，z，w]
    Matrix<double, 4, 1> x = svd.matrixV().col(3);      // 右奇异向量作为旋转四元数
    Quaterniond estimated_R(x);
    ric = estimated_R.toRotationMatrix().inverse();
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();
    // 倒数第二个奇异值，因为旋转是3个自由度，因此检查一下第三小的奇异值是否足够大，通常需要足够的运动激励才能保证得到没有奇异的解
    //至少迭代计算了WINDOW_SIZE次，且R的奇异值大于0.25才认为标定成功
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
    {
        calib_ric_result = ric;
        return true;
    }
    else
        return false;
}

Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    if (corres.size() >= 9)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        // 这里用的是相机坐标系，因此这个函数得到的也就是E矩阵
        cv::Mat E = cv::findFundamentalMat(ll, rr);
        cv::Mat_<double> R1, R2, t1, t2;
        // 本质矩阵的分解
        decomposeE(E, R1, R2, t1, t2);

        // 旋转矩阵的行列式应该是1,这里如果是-1就取一下反
        if (determinant(R1) + 1.0 < 1e-09)
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        // 本质矩阵分解完会有四组解，要检查一下哪个是合适的
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

        // 解出来的是R21

        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j); // 这里转换成R12
        return ans_R_eigen;
    }
    return Matrix3d::Identity();
}

/**
 * @brief 通过三角化来检查R t是否合理
 * 
 * @param[in] l l相机的观测
 * @param[in] r r相机的观测
 * @param[in] R 旋转矩阵
 * @param[in] t 位移
 * @return double 
 */
double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud;
    // 其中一帧设置为单位阵
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    // 第二帧就设置为R t对应的位姿
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    // 看一下opencv的定义
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++)
    {
        // 因为是齐次的，所以要求最后一维等于1
        double normal_factor = pointcloud.col(i).at<float>(3);
        // 得到在各自相机坐标系下的3d坐标
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        // 通过深度是否大于0来判断是否合理
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    return 1.0 * front_count / pointcloud.cols;
}

// 具体解法参考多视角几何
void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
