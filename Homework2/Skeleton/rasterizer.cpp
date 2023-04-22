// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <algorithm>
using namespace Eigen;

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(int x, int y, const Triangle& tri)
{
    // check if point p is in the triangle(2d) tri
    Vector3f p = {(float) x, (float) y, 0};
    Vector3f v_0 = { tri.v[0][0], tri.v[0][1], 0};
    Vector3f v_1 = { tri.v[1][0], tri.v[1][1], 0};
    Vector3f v_2 = { tri.v[2][0], tri.v[2][1], 0};
    // construct relative vector
    Vector3f v_0v_1 = v_1 - v_0, v_0p = p - v_0;
    Vector3f v_1v_2 = v_2 - v_1, v_1p = p - v_1;
    Vector3f v_2v_0 = v_0 - v_2, v_2p = p - v_2;
    // get the cross product
    Vector3f cp_0 = v_0v_1.cross(v_0p);
    Vector3f cp_1 = v_1v_2.cross(v_1p);
    Vector3f cp_2 = v_2v_0.cross(v_2p);
    // if the direction of all cross products follow +z, then p is in the tri
    bool right_side = cp_0[2] > 0 && cp_1[2] > 0 && cp_2[2] > 0;
    bool left_side = cp_0[2] < 0 && cp_1[2] < 0 && cp_2[2] < 0;
    return right_side || left_side;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t)
{
    auto v = t.toVector4();
    // Get the bounding box of a triangle(2d)
    // Using a trivial algorithm. for each given triangle, we find min_x, min_y, max_x, max_y
    // the bounding box can be [min_x, max_x] * [min_y, max_y]
    // A more advanced algorithm is so-called "Rotating Calipers". Check it on Google for more details.
    int min_x = 701, max_x = -1, min_y = 701, max_y = -1; // set the initial values by resolution 700x700
    // traverse vertices of the given triangle
    for (Vector4f& vertex : v)
    {
        float v_x = vertex[0], v_y = vertex[1];
        // scan and update related parameters
        min_x = v_x < (float) min_x ? ((int) v_x) - 1 : min_x;
        min_y = v_y < (float) min_y ? ((int) v_y) - 1 : min_y;
        max_x = v_x > (float) max_x ? ((int) v_x) + 1 : max_x;
        max_y = v_y > (float) max_y ? ((int) v_y) + 1 : max_y;
    }
    // scan bounded pixel points to check if it is in the triangle
    for (int x = min_x; x <= max_x; x++)
    {
        for (int y = min_y; y <= max_y; y++)
        {
            if (insideTriangle(x, y, t))
            {
                // get the depth interpolation
                auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
                float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                // do z-buffer algorithm
                int index = get_index(x, y);
                if (z_interpolated < depth_buf[index]) {
                    // update the current pixel's depth buffer
                    depth_buf[index] = z_interpolated;
                    // shade the current pixel
                    Vector3f pixel_point = {(float) x, (float) y, z_interpolated};
                    set_pixel(pixel_point, t.getColor());
                }
            }
        }
    }
}

void rst::rasterizer::set_model(const Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Vector3f& point, const Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on