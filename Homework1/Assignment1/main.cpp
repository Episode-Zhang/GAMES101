#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
using namespace Eigen;

constexpr double MY_PI = 3.1415926;

// move camera to eye position
Matrix4f get_view_matrix(Vector3f eye_pos)
{
    Matrix4f view = Matrix4f::Identity();
    return view;
}

Matrix4f get_camera_matrix(Vector3f eye_gaze, Vector3f eye_view_up, Vector3f eye_pos)
{
    Vector3f w = -eye_gaze * (1.0f / eye_gaze.norm());
    Vector3f u = eye_view_up.cross(w) * (1.0f / (eye_view_up.cross(w)).norm());
    Vector3f v = w.cross(u);
    Matrix4f camera_to_world {
            {u[0], v[0], w[0], eye_pos[0]},
            {u[1], v[1], w[1], eye_pos[1]},
            {u[2], v[2], w[2], eye_pos[2]},
            {0, 0, 0, 1},
    };
    return camera_to_world.inverse();
}

Matrix4f get_model_matrix(float rotation_angle)
{
    Matrix4f model = Matrix4f::Identity();
    /*
     * Since MVP transformation takes Model transformation as its first step,
     * considering the part of "Viewing Transformation" introduced in Tiger Book,
     * we know all transformations start with camera transformation, aka, in
     * fact we are using camera to observe the whole scene(or, world). So all
     * transformations we did should be *CAPTURED* in the *CAMERA WORLD*.
     * Mathematically saying, we should process all vertices' coordinates from
     * world coordinates into camera coordinates. Thus, here, in the function of
     * the model transformation, we do camera transformation.
     */
    // camera transformation
    // transform all triangle vertices coordinates into camera-coord system
    Vector3f eye_gaze = {0, 0, 1}; // eye gaze in +z direction
    Vector3f eye_view_up = { 0, 1, 0 }; // head-up pos in +y direction
    Vector3f eye_pos = {0, 0, 5}; // see main function
    Matrix4f to_cam_coord_mat = get_camera_matrix(eye_gaze, eye_view_up, eye_pos);
    // Model Transformation
    // using rad mode
    float rotation_angle_rad = rotation_angle / 180.0f * (float) MY_PI;
    // calc rotation mat
    Matrix4f rotation_mat {
        {cos(rotation_angle_rad), -sin(rotation_angle_rad), 0, 0},
        {sin(rotation_angle_rad), cos(rotation_angle_rad), 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1},
    };
    // merge
    model = rotation_mat * to_cam_coord_mat * model;
    return model;
}

Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    Matrix4f projection = Matrix4f::Identity();
    // calculate related parameters
    // Perspective Projection Mat
    float angle = eye_fov / 180 * (float) MY_PI;
    float t = fabs(zNear) * tan(angle / 2), b = -t;
    float r = (t - b) * aspect_ratio / 2.0f, l = -r;
    Matrix4f perspective_mat {
        {zNear, 0, 0, 0},
        {0, zNear, 0, 0},
        {0, 0, zNear + zFar, -zNear * zFar},
        {0, 0, 1, 0},
    };
    // Orthogonal Projection Mat
    Matrix4f orthogonal_mat {
        {2 / (r - l), 0, 0, -(l + r) / 2},
        {0, 2 / (t - b), 0, -(b + t) / 2},
        {0, 0, 2 / (zNear - zFar), -(zFar + zNear) / 2},
        {0, 0, 0, 1},
    };
    // left-hand coordinate system
    Matrix4f left_hand_mat {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, -1, 0},
        {0, 0, 0, 1},
    };
    // merge
    projection = left_hand_mat * orthogonal_mat * perspective_mat;
    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
    }

    rst::rasterizer r(700, 700);

    Vector3f eye_pos = {0, 0, 5};

    std::vector<Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
