// clang-format off
#include <iostream>
#include <opencv2/opencv.hpp>
#include "rasterizer.hpp"
#include "global.hpp"
#include "Triangle.hpp"

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_camera_matrix(Eigen::Vector3f eye_gaze, Eigen::Vector3f eye_view_up, Eigen::Vector3f eye_pos)
{
    Eigen::Vector3f w = -eye_gaze * (1.0f / eye_gaze.norm());
    Eigen::Vector3f u = eye_view_up.cross(w) * (1.0f / (eye_view_up.cross(w)).norm());
    Eigen::Vector3f v = w.cross(u);
    Eigen::Matrix4f camera_to_world {
            {u[0], v[0], w[0], eye_pos[0]},
            {u[1], v[1], w[1], eye_pos[1]},
            {u[2], v[2], w[2], eye_pos[2]},
            {0, 0, 0, 1},
    };
    return camera_to_world.inverse();
}

Eigen::Matrix4f get_camera_world_basis(Eigen::Vector3f eye_gaze, Eigen::Vector3f eye_view_up)
{
    Eigen::Vector3f w = -eye_gaze * (1.0f / eye_gaze.norm());
    Eigen::Vector3f u = eye_view_up.cross(w) * (1.0f / (eye_view_up.cross(w)).norm());
    Eigen::Vector3f v = w.cross(u);
    Eigen::Matrix4f camera_world_basis {
            {u[0], v[0], w[0], 0},
            {u[1], v[1], w[1], 0},
            {u[2], v[2], w[2], 0},
            {0, 0, 0, 1},
    };
    return camera_world_basis;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    // calculate related parameters
    // Camera Transformation Matrix
    // Since zNear and zFar are both positive
    // We need to make it gaze to -z direction
    // To achieve this, we do camera transformation
    Eigen::Vector3f eye_gaze(0, 0, 1); // eye look up +z direction actually
    Eigen::Vector3f eye_view_up(0, 1, 0);
    Eigen::Vector3f eye_pos(0, 0, 5); // eye_pos is (0, 0, 5), see main() function
    Eigen::Matrix4f camera_mat = get_camera_matrix(eye_gaze, eye_view_up, eye_pos);
    // Perspective Projection Mat
    float angle = eye_fov / 180 * (float) MY_PI;
    float t = fabs(zNear) * tan(angle / 2), b = -t;
    float r = (t - b) * aspect_ratio / 2.0f, l = -r;
    Eigen::Matrix4f perspective_mat {
            {zNear, 0, 0, 0},
            {0, zNear, 0, 0},
            {0, 0, zNear + zFar, -zNear * zFar},
            {0, 0, 1, 0},
    };
    // Orthogonal Projection Mat
    Eigen::Matrix4f orthogonal_mat {
            {2 / (r - l), 0, 0, -(l + r) / 2},
            {0, 2 / (t - b), 0, -(b + t) / 2},
            {0, 0, 2 / (zNear - zFar), -(zFar + zNear) / 2},
            {0, 0, 0, 1},
    };
    // merge
    projection = orthogonal_mat * perspective_mat * camera_mat;
    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc == 2)
    {
        command_line = true;
        filename = std::string(argv[1]);
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0,0,5};


    std::vector<Eigen::Vector3f> pos
            {
                    {2, 0, -2},
                    {0, 2, -2},
                    {-2, 0, -2},
                    {3.5, -1, -5},
                    {2.5, 1.5, -5},
                    {-1, 0.5, -5}
            };

    std::vector<Eigen::Vector3i> ind
            {
                    {0, 1, 2},
                    {3, 4, 5}
            };

    std::vector<Eigen::Vector3f> cols
            {
                    {217.0, 238.0, 185.0},
                    {217.0, 238.0, 185.0},
                    {217.0, 238.0, 185.0},
                    {185.0, 217.0, 238.0},
                    {185.0, 217.0, 238.0},
                    {185.0, 217.0, 238.0}
            };

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);
    auto col_id = r.load_colors(cols);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';
    }

    return 0;
}
// clang-format on