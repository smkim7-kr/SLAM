#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>
#include <iostream>
#include <random>

// Vertex class for ellipse parameters (a, b, cx, cy, theta)
class VertexEllipse : public g2o::BaseVertex<5, Eigen::Vector5d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override {
        _estimate << 1.0, 1.0, 0.0, 0.0, 0.0;
    }

    virtual void oplusImpl(const double* update) override {
        _estimate += Eigen::Map<const Eigen::Vector5d>(update);
    }

    virtual bool read(std::istream& /*is*/) override { return true; }
    virtual bool write(std::ostream& /*os*/) const override { return true; }
};

// Edge class for ellipse fitting
class EdgeEllipsePoint : public g2o::BaseUnaryEdge<1, Eigen::Vector2d, VertexEllipse> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeEllipsePoint() : BaseUnaryEdge<1, Eigen::Vector2d, VertexEllipse>() {}

    virtual void computeError() override {
        const VertexEllipse* v = static_cast<const VertexEllipse*>(_vertices[0]);
        const Eigen::Vector5d& params = v->estimate();
        
        double a = params[0];
        double b = params[1];
        double cx = params[2];
        double cy = params[3];
        double theta = params[4];

        // Transform point to ellipse coordinate system
        Eigen::Vector2d p = _measurement;
        p -= Eigen::Vector2d(cx, cy);
        
        // Rotate point
        double cos_t = std::cos(-theta);
        double sin_t = std::sin(-theta);
        double x_rot = cos_t * p.x() - sin_t * p.y();
        double y_rot = sin_t * p.x() + cos_t * p.y();

        // Compute error (distance from point to ellipse)
        _error[0] = std::pow(x_rot/a, 2) + std::pow(y_rot/b, 2) - 1.0;
    }

    virtual bool read(std::istream& /*is*/) override { return true; }
    virtual bool write(std::ostream& /*os*/) const override { return true; }
};

// Generate random points on ellipse with noise
std::vector<Eigen::Vector2d> generateEllipsePoints(double a, double b, double cx, double cy, 
                                                  double theta, int n_points, double noise_std) {
    std::vector<Eigen::Vector2d> points;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, noise_std);
    
    for (int i = 0; i < n_points; ++i) {
        double t = 2 * M_PI * i / n_points;
        double x = a * std::cos(t);
        double y = b * std::sin(t);
        
        // Rotate
        double cos_t = std::cos(theta);
        double sin_t = std::sin(theta);
        double x_rot = cos_t * x - sin_t * y;
        double y_rot = sin_t * x + cos_t * y;
        
        // Translate and add noise
        x_rot += cx + noise(gen);
        y_rot += cy + noise(gen);
        
        points.push_back(Eigen::Vector2d(x_rot, y_rot));
    }
    return points;
}

int main() {
    // True ellipse parameters (a, b, cx, cy, theta)
    double a_true = 5.0;
    double b_true = 3.0;
    double cx_true = 1.0;
    double cy_true = 2.0;
    double theta_true = M_PI / 6.0;

    // Generate noisy data points
    std::vector<Eigen::Vector2d> points = generateEllipsePoints(
        a_true, b_true, cx_true, cy_true, theta_true, 100, 0.1);

    // Setup g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<5, 1>> BlockSolver_5_1;
    typedef g2o::LinearSolverDense<BlockSolver_5_1::PoseMatrixType> LinearSolver;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolver_5_1>(g2o::make_unique<LinearSolver>()));
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // Add vertex
    VertexEllipse* v = new VertexEllipse();
    v->setId(0);
    v->setEstimate(Eigen::Vector5d(4.0, 2.5, 0.8, 1.8, M_PI/7.0)); // Initial guess
    optimizer.addVertex(v);

    // Add edges
    for (const auto& p : points) {
        EdgeEllipsePoint* e = new EdgeEllipsePoint();
        e->setVertex(0, v);
        e->setMeasurement(p);
        e->setInformation(Eigen::Matrix<double,1,1>::Identity());
        optimizer.addEdge(e);
    }

    // Optimize
    optimizer.initializeOptimization();
    optimizer.optimize(50);

    // Print results
    std::cout << "True parameters: " << a_true << ", " << b_true << ", " 
              << cx_true << ", " << cy_true << ", " << theta_true << std::endl;
    std::cout << "Estimated parameters: " << v->estimate().transpose() << std::endl;

    return 0;
}