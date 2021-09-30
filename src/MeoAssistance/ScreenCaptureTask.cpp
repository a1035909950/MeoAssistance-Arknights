#include "ScreenCaptureTask.h"

#include <opencv2/opencv.hpp>

#include "AsstDef.h"
#include "AsstAux.h"
#include "Logger.hpp"
#include "Configer.h"
#include "InfrastConfiger.h"
#include "WinMacro.h"

bool asst::ScreenCaptureTask::run()
{
    if (m_controller_ptr == nullptr
        || m_identify_ptr == nullptr)
    {
        m_callback(AsstMsg::PtrIsNull, json::value(), m_callback_arg);
        return false;
    }

    //static const std::string dirname = GetCurrentDir() + "template\\";
    //return print_window(dirname, false);

    return cap_opers_name_in_infrast();
}

bool asst::ScreenCaptureTask::cap_opers_name_in_list()
{
    DebugTraceFunction;

    cv::Mat image = m_controller_ptr->get_image();

    std::vector<Rect> unrecognized;

    auto identify_cropped = [&](cv::Mat cropped, int y_offset)
        -> std::vector<TextArea> {
        // ocr库，单色图片识别效果好很多；但是只接受三通道的图片，所以这里转两次，送进去单色的、三通道的图片
        cv::cvtColor(cropped, cropped, cv::COLOR_BGR2GRAY);
        cv::cvtColor(cropped, cropped, cv::COLOR_GRAY2BGR);

        std::vector<TextArea> all_text_area = ocr_detect(cropped);	// 所有文字
        // 因为图片是裁剪过的，所以对应原图的坐标要加上裁剪的参数
        for (TextArea& textarea : all_text_area) {
            textarea.rect.y += y_offset;
        }
        // 过滤出所有的干员名
        std::vector<TextArea> opers_name = text_match(
            all_text_area,
            InfrastConfiger::get_instance().m_all_opers_name,
            Configer::get_instance().m_infrast_ocr_replace);

        // 把这一块涂黑，避免后面被特征检测的误识别了
        //for (const TextArea& textarea : upper_part_names) {
        //    cv::Rect rect(textarea.rect.x, textarea.rect.y - cropped_upper_y, textarea.rect.width, textarea.rect.height);
        //    // 这里是转过灰度图再转回来的，相当于深拷贝，不会影响原图
        //    cv::rectangle(upper_part_name_image, rect, cv::Scalar(0, 0, 0), -1);
        //}
        if (opers_name.empty()) {
            // TODO:报错！这一行一个都没识别出来
        }
        const Rect& refer_rect = opers_name.front().rect;
        constexpr int ReferOffset = Configer::WindowWidthDefault * -0.01;
        int refer_pos = refer_rect.x + refer_rect.width + ReferOffset;
        constexpr int Spacing = Configer::WindowWidthDefault * 0.124; // 0.1125;//0.124;
        std::unordered_set<int> pos_set;
        pos_set.emplace(refer_pos);
        int left_pos = refer_pos;
        while (left_pos > Spacing) {
            left_pos -= Spacing;
            pos_set.emplace(left_pos);
        }
        int right_pos = refer_pos;
        while (right_pos < Configer::WindowWidthDefault) {
            pos_set.emplace(right_pos);
            right_pos += Spacing;
        }

        for (const TextArea& ta : opers_name) {
            auto iter = std::find_if(pos_set.begin(), pos_set.end(),
                [&](int pos) -> bool {
                    return ta.rect.x < pos && (ta.rect.x + ta.rect.width) > pos;
                });
            if (iter != pos_set.end()) {
                pos_set.erase(iter);
            }
        }
        constexpr int PosLeftOffset = Configer::WindowWidthDefault * 0.05;
        constexpr int OperatorRectWidth = Configer::WindowWidthDefault * 0.065;
        for (int pos : pos_set) {
            int x = pos - PosLeftOffset;
            if (x < 0) {
                continue;
            }
            if (x + OperatorRectWidth > Configer::WindowWidthDefault) {
                continue;
            }
            unrecognized.emplace_back(x, refer_rect.y, OperatorRectWidth, refer_rect.height);
        }
        return opers_name;
    };

    m_cropped_height_ratio = 0.043;
    m_cropped_upper_y_ratio = 0.480;
    m_cropped_lower_y_ratio = 0.923;

    // 裁剪出来干员名的一个长条形图片，没必要把整张图片送去识别
    int cropped_height = image.rows * m_cropped_height_ratio;
    int cropped_upper_y = image.rows * m_cropped_upper_y_ratio;
    int cropped_lower_y = image.rows * m_cropped_lower_y_ratio;

    // 识别上半部分的干员
    cv::Mat upper_part_name_image = image(cv::Rect(0, cropped_upper_y, image.cols, cropped_height));
    auto upper_part_names = identify_cropped(upper_part_name_image, cropped_upper_y);

    // 下半部分的干员
    cv::Mat lower_part_name_image = image(cv::Rect(0, cropped_lower_y, image.cols, cropped_height));
    auto lower_part_names = identify_cropped(lower_part_name_image, cropped_lower_y);

    std::vector<TextArea> all_opers_textarea = std::move(upper_part_names);
    all_opers_textarea.insert(all_opers_textarea.end(),
        std::make_move_iterator(lower_part_names.begin()),
        std::make_move_iterator(lower_part_names.end()));

#ifdef LOG_TRACE
    cv::Mat draw_image = image;
    for (const auto& textarea : all_opers_textarea) {
        std::string filename = GetCurrentDir() + "temp\\" + Utf8ToGbk(textarea.text) + ".png";
        cv::Rect save_rect = make_rect<cv::Rect>(textarea.rect);
        if (save_rect.y < 500) {
            save_rect.y = 346;
        }
        else {
            save_rect.y = 664;
        }
        save_rect.height = 30;
        cv::imwrite(filename, image(save_rect));
        cv::rectangle(draw_image, make_rect<cv::Rect>(textarea.rect), cv::Scalar(0, 0, 255));
    }
    static int index = 0;
    for (const Rect& rect : unrecognized) {
        std::string filename = GetCurrentDir() + "temp\\" + std::to_string(index++) + ".png";
        cv::Rect save_rect = make_rect<cv::Rect>(rect);
        if (save_rect.y < 500) {
            save_rect.y = 346;
        }
        else {
            save_rect.y = 664;
        }
        save_rect.height = 30;
        if (save_rect.x + save_rect.width >= Configer::WindowWidthDefault) {
            save_rect.width = Configer::WindowWidthDefault - save_rect.x;
        }
        cv::imwrite(filename, image(save_rect));
        cv::rectangle(draw_image, make_rect<cv::Rect>(rect), cv::Scalar(255, 0, 0), 3);
    }
    cv::imwrite(GetCurrentDir() + "temp\\draw.png", draw_image);
#endif
    return true;
}

bool asst::ScreenCaptureTask::cap_opers_name_in_infrast()
{
    DebugTraceFunction;

    cv::Mat image = m_controller_ptr->get_image();

    std::vector<Rect> unrecognized;

    constexpr int InfrastInfoWidth = 400;

    auto identify_cropped = [&](cv::Mat cropped, int y_offset)
        -> std::vector<TextArea> {
        // ocr库，单色图片识别效果好很多；但是只接受三通道的图片，所以这里转两次，送进去单色的、三通道的图片
        cv::cvtColor(cropped, cropped, cv::COLOR_BGR2GRAY);
        cv::cvtColor(cropped, cropped, cv::COLOR_GRAY2BGR);

        std::vector<TextArea> all_text_area = ocr_detect(cropped);	// 所有文字
        // 因为图片是裁剪过的，所以对应原图的坐标要加上裁剪的参数
        for (TextArea& textarea : all_text_area) {
            textarea.rect.y += y_offset;
            textarea.rect.x += InfrastInfoWidth;
        }
        // 过滤出所有的干员名
        std::vector<TextArea> opers_name = text_match(
            all_text_area,
            InfrastConfiger::get_instance().m_all_opers_name,
            Configer::get_instance().m_infrast_ocr_replace);

        // 把这一块涂黑，避免后面被特征检测的误识别了
        //for (const TextArea& textarea : upper_part_names) {
        //    cv::Rect rect(textarea.rect.x, textarea.rect.y - cropped_upper_y, textarea.rect.width, textarea.rect.height);
        //    // 这里是转过灰度图再转回来的，相当于深拷贝，不会影响原图
        //    cv::rectangle(upper_part_name_image, rect, cv::Scalar(0, 0, 0), -1);
        //}
        if (opers_name.empty()) {
            // TODO:报错！这一行一个都没识别出来
        }
        const Rect& refer_rect = opers_name.front().rect;
        constexpr int ReferOffset = Configer::WindowWidthDefault * -0.01;
        int refer_pos = refer_rect.x + refer_rect.width + ReferOffset;
        constexpr int Spacing = Configer::WindowWidthDefault * 0.1125; // 0.1125;//0.124;
        std::unordered_set<int> pos_set;
        pos_set.emplace(refer_pos);
        int left_pos = refer_pos;
        while (left_pos > Spacing) {
            left_pos -= Spacing;
            pos_set.emplace(left_pos);
        }
        int right_pos = refer_pos;
        while (right_pos < Configer::WindowWidthDefault) {
            pos_set.emplace(right_pos);
            right_pos += Spacing;
        }

        for (const TextArea& ta : opers_name) {
            auto iter = std::find_if(pos_set.begin(), pos_set.end(),
                [&](int pos) -> bool {
                    return ta.rect.x < pos && (ta.rect.x + ta.rect.width) > pos;
                });
            if (iter != pos_set.end()) {
                pos_set.erase(iter);
            }
        }
        constexpr int PosLeftOffset = Configer::WindowWidthDefault * 0.055;
        constexpr int OperatorRectWidth = Configer::WindowWidthDefault * 0.065;
        for (int pos : pos_set) {
            int x = pos - PosLeftOffset;
            if (x < 0) {
                continue;
            }
            if (x + OperatorRectWidth > Configer::WindowWidthDefault) {
                continue;
            }
            unrecognized.emplace_back(x, refer_rect.y, OperatorRectWidth, refer_rect.height);
        }
        return opers_name;
    };

    // 裁剪出来干员名的一个长条形图片，没必要把整张图片送去识别
    int cropped_height = image.rows * m_cropped_height_ratio;
    int cropped_upper_y = image.rows * m_cropped_upper_y_ratio;
    int cropped_lower_y = image.rows * m_cropped_lower_y_ratio;

    // 识别上半部分的干员
    cv::Mat upper_part_name_image = image(cv::Rect(InfrastInfoWidth, cropped_upper_y, image.cols - InfrastInfoWidth, cropped_height));
    auto upper_part_names = identify_cropped(upper_part_name_image, cropped_upper_y);

    // 下半部分的干员
    cv::Mat lower_part_name_image = image(cv::Rect(InfrastInfoWidth, cropped_lower_y, image.cols - InfrastInfoWidth, cropped_height));
    auto lower_part_names = identify_cropped(lower_part_name_image, cropped_lower_y);

    std::vector<TextArea> all_opers_textarea = std::move(upper_part_names);
    all_opers_textarea.insert(all_opers_textarea.end(),
        std::make_move_iterator(lower_part_names.begin()),
        std::make_move_iterator(lower_part_names.end()));

#ifdef LOG_TRACE
    cv::Mat draw_image = image;
    for (const auto& textarea : all_opers_textarea) {
        std::string filename = GetCurrentDir() + "temp\\" + Utf8ToGbk(textarea.text) + ".png";
        cv::Rect save_rect = make_rect<cv::Rect>(textarea.rect);
        if (save_rect.y < 500) {
            save_rect.y = 318;
        }
        else {
            save_rect.y = 599;
        }
        save_rect.height = 30;
        cv::imwrite(filename, image(save_rect));
        cv::rectangle(draw_image, make_rect<cv::Rect>(textarea.rect), cv::Scalar(0, 0, 255));
    }
    static int index = 0;
    for (const Rect& rect : unrecognized) {
        std::string filename = GetCurrentDir() + "temp\\" + std::to_string(index++) + ".png";
        cv::Rect save_rect = make_rect<cv::Rect>(rect);
        if (save_rect.x < InfrastInfoWidth) {
            continue;
        }
        if (save_rect.y < 500) {
            save_rect.y = 318;
        }
        else {
            save_rect.y = 599;
        }
        save_rect.height = 30;
        if (save_rect.x + save_rect.width >= Configer::WindowWidthDefault) {
            save_rect.width = Configer::WindowWidthDefault - save_rect.x;
        }
        cv::imwrite(filename, image(save_rect));
        cv::rectangle(draw_image, make_rect<cv::Rect>(save_rect), cv::Scalar(255, 0, 0), 3);
    }
    cv::imwrite(GetCurrentDir() + "temp\\draw.png", draw_image);
#endif
    return true;
}